import torch
import torch.nn as nn
from datasets import load_dataset
from dataclasses import dataclass
from utils import tokenize_and_concatenate
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from pandas import DataFrame
import gc
import wandb
from functools import partial
from einops import *



@dataclass
class MidcoderConfig():
    context_length = 128
    mid_dim = 'd_model'
    mse_out_param = 1

    seed: int = 42
    lr = 1e-3
    weight_decay = 1e-3
    batch_size = 32
    steps_per_epoch = 10
    train_tokens = 100_000_000
    log = True
    wandb_project = 'midcoder'
    device = 'cuda'  

    saved_features = 2000
    
class Midcoder(nn.Module):
    # finds the mid vectors (pre-ReLU feature vectors) 
    # for a previously trained transcoder
    # based on jacobdunefsky/transcoder_circuits and pchlenski/gpt2-transcoders

    def __init__(self, models, layer, cfg: MidcoderConfig):
        super().__init__()
        # models: dict of {'model': HookedTransformer, 'transcoder':SparseAutoencoder} to avoid parameters being added
        self.models = models
        self.layer = layer
        self.cfg = cfg
        self.device = cfg.device
        self.pre_hook_point = models['model'].blocks[layer].ln2.hook_normalized
        self.post_hook_point = models['model'].blocks[layer].hook_mlp_out

        torch.manual_seed(self.cfg.seed)

        # transcoder weights
        self.b_dec = models['transcoder'].b_dec.clone().detach()
        self.W_enc = models['transcoder'].W_enc.clone().detach()
        self.b_enc = models['transcoder'].b_enc.clone().detach()
        self.W_dec = models['transcoder'].W_dec.clone().detach()
        self.b_dec_out = models['transcoder'].b_dec_out.clone().detach()

        self.W_in = models['model'].blocks[self.layer].mlp.W_in.clone().detach()
        self.b_in = models['model'].blocks[self.layer].mlp.b_in.clone().detach()
        self.W_out = models['model'].blocks[self.layer].mlp.W_out.clone().detach()
        self.b_out = models['model'].blocks[self.layer].mlp.b_out.clone().detach()
        
        n_feat = self.W_enc.shape[1]
        d_model, d_mlp = models['model'].blocks[layer].mlp.W_in.shape

        if self.cfg.mid_dim == 'd_model':
            self.W_mid = nn.Parameter(
                # self.W_dec @ torch.linalg.pinv(self.W_in @ self.W_out),
                # torch.nn.init.kaiming_uniform_(torch.empty(n_feat, d_model, device=self.device)),
                torch.zeros((n_feat, d_model), device=self.device),
                requires_grad=True
            )
            self.b_mid = nn.Parameter(
                            torch.zeros(d_model,  device=self.device), 
                            requires_grad=True)
        elif self.cfg.mid_dim == 'd_mlp':
            self.W_mid = nn.Parameter(
                # self.W_dec @ torch.linalg.pinv(self.W_in @ self.W_out),
                # torch.nn.init.kaiming_uniform_(torch.empty(n_feat, d_model, device=self.device)),
                torch.zeros((n_feat, d_mlp), device=self.device),
                requires_grad=True
            )
            self.b_mid = nn.Parameter(
                            torch.zeros(d_mlp,  device=self.device), 
                            requires_grad=True)

        self.b_mid_out = nn.Parameter(torch.zeros(d_model, device=self.device),
                                        requires_grad=True)

        self.initialize_saved_values(n_feat, d_model, d_mlp)

    def forward(self, inputs):
        #inputs: input into MLP of shape (batch, pos, d_model)
        x = inputs.to(self.device)
        x = x - self.b_dec
        acts = torch.nn.functional.relu(x @ self.W_enc + self.b_enc) #activations

        if self.cfg.mid_dim == 'd_model':
            mid = acts @ self.W_mid + self.b_mid
            mid = mid @ self.W_in + self.b_in
        elif self.cfg.mid_dim == 'd_mlp':
            mid = acts @ self.W_mid + self.b_mid
        gate = (mid > 0).float()
        # mlp_out = relu_out @ self.W_out + self.b_out
        # mlp_out = (gate * mid) @ self.W_out + self.b_mid_out
        mlp_out = (gate * mid) @ self.W_out + self.b_mid_out
        return mlp_out, gate, mid, acts
    
    def get_inputs_outputs(self, token_array):
        #x: input to model of shape (batch, pos)
        with torch.no_grad():
            _, cache = self.models['model'].run_with_cache(token_array, stop_at_layer=self.layer+1, 
                                                names_filter=[self.pre_hook_point.name,
                                                              self.post_hook_point.name])
            inputs = cache[self.pre_hook_point.name]
            outputs = cache[self.post_hook_point.name]
        return inputs, outputs
    
    def step(self, batch):
        inputs, outputs = self.get_inputs_outputs(batch)
        mlp_out, gate, mid, acts = self.forward(inputs)
        mlp_out_trans = acts @ self.W_dec + self.b_dec_out

        if self.training:
            self.update_saved_values(gate, acts, mid, inputs, outputs)

        mse_mid = nn.MSELoss()(mid, inputs @ self.W_in + self.b_in)
        mse_out = nn.MSELoss()(mlp_out, outputs)
        mse_out_trans = nn.MSELoss()(mlp_out_trans, outputs)

        loss = mse_mid + self.cfg.mse_out_param * mse_out
        # scale = (outputs**2).mean()
        # mse_loss_norm = mse_loss/scale
        return loss, mse_mid, mse_out, mse_out_trans

    def get_dataset(self):
        dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True,
                                trust_remote_code=True)
        dataset = dataset.shuffle(seed=self.cfg.seed, buffer_size=10_000)
        tokenized_dataset = tokenize_and_concatenate(dataset, self.models['model'].tokenizer, 
                            max_length=self.cfg.context_length, streaming=True
                            )
        return tokenized_dataset

    def get_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.cfg.batch_size
        dataset = self.get_dataset()
        return iter(DataLoader(dataset, batch_size=batch_size))
    
    def fit(self):
        if self.cfg.log: wandb.init(project=self.cfg.wandb_project, config=self.cfg)

        optimizer = AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        num_epochs = int(self.cfg.train_tokens/(self.cfg.context_length * self.cfg.batch_size * self.cfg.steps_per_epoch)) + 1
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        dataloader = self.get_dataloader()
        
        pbar = tqdm(range(num_epochs))
        history = []
        for _ in pbar:
            epoch = []
            for step in range(self.cfg.steps_per_epoch):
                batch = next(dataloader)['tokens']
                loss, mse_mid, mse_out, mse_out_trans = self.train().step(batch)
                epoch += [(loss.item(), mse_mid.item(), mse_out.item(), mse_out_trans.item())]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            metrics = {
                "loss": sum([step[0] for step in epoch]) / len(epoch),
                "mse_mid": sum([step[1] for step in epoch]) / len(epoch),
                "mse_out": sum([step[2] for  step in epoch]) / len(epoch),
                "mse_out_trans": sum([step[3] for  step in epoch]) / len(epoch),
                "lr": scheduler.get_last_lr()[0]
            }
            if self.cfg.log: wandb.log(metrics)
            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3e}" for k, v in metrics.items()))

            self.clear_cache()
            
        if self.cfg.log: wandb.finish()
        return

    def evaluate(self, tokens=2e6):
        dataloader = self.get_dataloader()
        steps = int(tokens/(self.cfg.context_length * self.cfg.batch_size))+1
        pbar = tqdm(range(steps))

        self.set_avg_input_output_quants()
        
        epoch = []
        for _ in pbar:
            with torch.no_grad():
                batch = next(dataloader)['tokens']
                inputs, outputs = self.get_inputs_outputs(batch)
                # self.add_avg_input_output_quants(inputs, outputs)

                mlp_out, relu_out, mid, acts = self.forward(inputs)
                mlp_out_trans = acts @ self.W_dec + self.b_dec_out
                loss, mid_loss, trans_loss = self.get_reconstruction_losses(batch,
                                                                            mlp_out,
                                                                            mlp_out_trans
                                                                            )
                mse_mid = nn.MSELoss()(mid, inputs @ self.W_in + self.b_in)
                mse_out = nn.MSELoss()(mlp_out, outputs)
                mse_out_trans = nn.MSELoss()(mlp_out_trans, outputs)

                epoch += [(loss.item(), mid_loss.item(), trans_loss.item(),
                          mse_mid.item(), mse_out.item(), mse_out_trans)]
                self.clear_cache()

        self.normalize_input_output_quants()
        metrics = {
            "ce_loss": sum([step[0] for step in epoch]) / len(epoch),
            "ce_loss_mid": sum([step[1] for step in epoch]) / len(epoch),
            "ce_loss_trans": sum([step[2] for step in epoch]) / len(epoch),
            "mse_mid": sum([step[3] for step in epoch]) / len(epoch),
            "mse_out": sum([step[4] for step in epoch]) / len(epoch),
            "mse_out_trans": sum([step[5] for step in epoch]) / len(epoch),
        }
        return metrics

    def get_reconstruction_losses(self, batch, mid_out, trans_out):
        with torch.no_grad():
            hook_point = self.post_hook_point.name
            loss = self.models['model'](batch, return_type = "loss").detach().mean()
            self.models['model'].reset_hooks()
            self.clear_cache()

            def replacement_hook(activations, hook):
                return mid_out
            mid_loss = self.models['model'].run_with_hooks(
                batch,
                return_type="loss",
                fwd_hooks=[(hook_point, partial(replacement_hook))],
                clear_contexts = True
            ).detach().mean()
            self.models['model'].reset_hooks()
            self.clear_cache()

            def replacement_hook(activations, hook):
                return trans_out
            trans_loss = self.models['model'].run_with_hooks(
                batch,
                return_type="loss",
                fwd_hooks=[(hook_point, partial(replacement_hook))],
                clear_contexts = True
            ).detach().mean()
            self.models['model'].reset_hooks()

            self.clear_cache()

        return loss, mid_loss, trans_loss


    def initialize_saved_values(self, n_feat, d_model, d_mlp):
        save_feat = self.cfg.saved_features
        self.feat_ids = nn.Parameter(
                            torch.randint(n_feat, size=(save_feat,), device=self.device),
                            requires_grad = False)

        self.act_gate_sum = nn.Parameter(
                                    torch.zeros((save_feat, d_mlp), device=self.device),
                                    requires_grad = False)
        self.act_sum = nn.Parameter(
                                    torch.zeros((save_feat), device=self.device),
                                    requires_grad = False)
        self.act_cov_sum = nn.Parameter(
                                    torch.zeros((save_feat, save_feat), device=self.device),
                                    requires_grad = False)

        self.act_cov_counts = nn.Parameter(
                                    torch.zeros((save_feat, save_feat), device=self.device),
                                    requires_grad = False)

        self.act_sq_gate_sum = nn.Parameter(
                                    torch.zeros((save_feat, d_mlp), device=self.device),
                                    requires_grad = False)
        self.act_sq_sum = nn.Parameter(
                                    torch.zeros((save_feat), device=self.device),
                                    requires_grad = False)

        self.mid_sum = nn.Parameter(
                                    torch.zeros((d_mlp), device=self.device),
                                    requires_grad = False)

        self.gate_mid_sum = nn.Parameter(
                                    torch.zeros((d_mlp), device=self.device),
                                    requires_grad = False)
        
        self.gate_act_counts = nn.Parameter(
                                    torch.zeros((save_feat, d_mlp), device=self.device),
                                    requires_grad = False)
        
        self.gate_counts = nn.Parameter(torch.zeros((d_mlp), device=self.device), requires_grad = False)
        
        self.act_counts = nn.Parameter(torch.zeros((save_feat), device=self.device), requires_grad = False)
        
        self.counts = nn.Parameter(torch.zeros((1), device=self.device), requires_grad = False)
        
        
        
        self.pre_relu_sum = nn.Parameter(
                                    torch.zeros((d_mlp), device=self.device),
                                    requires_grad = False)
        self.pre_relu_sum_no_act = nn.Parameter(
                                    torch.zeros((save_feat, d_mlp), device=self.device),
                                    requires_grad = False)
        self.pre_relu_cov_sum = nn.Parameter(
                                    torch.zeros((d_mlp, d_mlp), device=self.device),
                                    requires_grad = False)

        self.pre_relu_cube_sum = nn.Parameter(
                                    torch.zeros((d_mlp), device=self.device),
                                    requires_grad = False)
        self.mlp_out_sum = nn.Parameter(
                                    torch.zeros((d_model), device=self.device),
                                    requires_grad = False)
        self.mlp_out_sum_no_act = nn.Parameter(
                                    torch.zeros((save_feat, d_model), device=self.device),
                                    requires_grad = False)
        self.mlp_out_cov_sum = nn.Parameter(
                                    torch.zeros((d_model, d_model), device=self.device),
                                    requires_grad = False)

    @torch.no_grad()
    def update_saved_values(self, gate, acts, mid, inputs, outputs):
        # relu_out: (batch, pos, d_mlp)
        # acts: (batch, pos, n_feat)
        acts = acts[:,:,self.feat_ids]
        acts_bool = (acts > 0).float()

        self.act_gate_sum += einsum(gate, acts,"b pos d, b pos f -> f d")
        self.act_sum += einsum(acts, "b pos f -> f")
        self.act_cov_sum += einsum(acts, acts, "b pos f1, b pos f2 -> f1 f2")
        self.act_cov_counts += einsum(acts_bool, acts_bool, "b pos f1, b pos f2 -> f1 f2")

        self.act_sq_gate_sum += einsum(gate, acts**2,"b pos d, b pos f -> f d")
        self.act_sq_sum += einsum(acts**2, "b pos f -> f")

        self.gate_act_counts += einsum(gate, acts_bool,"b pos d, b pos f -> f d")
        self.gate_counts += einsum(gate, "b pos f -> f")
        self.act_counts += einsum(acts_bool, "b pos f -> f")

        self.mid_sum += einsum(mid, "b pos d -> d")
        self.gate_mid_sum += einsum(gate, mid, "b pos d, b pos d -> d")

        batch, pos, d_mlp = gate.shape
        self.counts += batch * pos

        pre_relu = inputs @ self.W_in + self.b_in
        self.pre_relu_sum += einsum(pre_relu, "b pos d -> d")
        self.pre_relu_sum_no_act += einsum(pre_relu, acts_bool, "b pos d, b pos f-> f d")

        self.pre_relu_cov_sum += einsum(pre_relu, pre_relu, "b pos d1, b pos d2 -> d1 d2")
        self.pre_relu_cube_sum += einsum(pre_relu**3, "b pos d -> d")

        self.mlp_out_sum += einsum(outputs, "b pos d -> d")
        self.mlp_out_sum_no_act += einsum(outputs, acts_bool, "b pos d, b pos f-> f d")
        self.mlp_out_cov_sum += einsum(outputs, outputs, "b pos d1, b pos d2 -> d1 d2")

    # def set_avg_input_output_quants(self):
    #     d_model, d_mlp = self.W_in.shape
    #     self.pre_relu_mean = nn.Parameter(
    #                                 torch.zeros((d_mlp), device=self.device),
    #                                 requires_grad = False)
    #     self.pre_relu_cov = nn.Parameter(
    #                                 torch.zeros((d_mlp, d_mlp), device=self.device),
    #                                 requires_grad = False)

    #     self.pre_relu_cube = nn.Parameter(
    #                                 torch.zeros((d_mlp), device=self.device),
    #                                 requires_grad = False)
    #     self.mlp_out_mean = nn.Parameter(
    #                                 torch.zeros((d_model), device=self.device),
    #                                 requires_grad = False)
    #     self.mlp_out_cov = nn.Parameter(
    #                                 torch.zeros((d_model, d_model), device=self.device),
    #                                 requires_grad = False)

    #     self.quant_count = nn.Parameter(
    #                                 torch.zeros((1), device=self.device),
    #                                 requires_grad = False)

    # @torch.no_grad()
    # def add_avg_input_output_quants(self, inputs, outputs):
    #     pre_relu = inputs @ self.W_in + self.b_in
    #     self.pre_relu_mean += einsum(pre_relu, "b pos d -> d")
    #     self.pre_relu_cov += einsum(pre_relu, pre_relu, "b pos d1, b pos d2 -> d1 d2")
    #     self.pre_relu_cube += einsum(pre_relu**3, "b pos d -> d")

    #     self.mlp_out_mean += einsum(outputs, "b pos d -> d")
    #     self.mlp_out_cov += einsum(outputs, outputs, "b pos d1, b pos d2 -> d1 d2")
        
    #     batch, pos, d_mlp = pre_relu.shape
    #     self.quant_count += batch * pos

    # def normalize_input_output_quants(self):
    #     self.pre_relu_mean /= self.quant_count
    #     self.pre_relu_cov /= self.quant_count
    #     self.pre_relu_cube /= self.quant_count
    #     self.mlp_out_mean /= self.quant_count
    #     self.mlp_out_cov /= self.quant_count

        
    def save_weights(self, path):
        data = {'weights': self.state_dict(), 
                'config': self.cfg}
        torch.save(data, path)

    def load_weights(self, path, **kwargs):
        #assumes weights are first item in list
        data = torch.load(path, map_location=self.device)
        self.load_state_dict(data['weights'], **kwargs)

    def clear_cache(self):
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()