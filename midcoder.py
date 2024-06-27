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


@dataclass
class MidcoderConfig():
    context_length = 128

    seed: int = 42
    lr = 1e-3
    weight_decay = 1e-3
    batch_size = 32
    steps_per_epoch = 10
    epochs = 100
    log = True
    wandb_project = 'midcoder'

    device = 'cuda'
    
    
class Midcoder(nn.Module):
    # finds the mid vectors (pre-ReLU feature vectors) 
    # for a previously trained transcoder
    # based on jacobdunefsky/transcoder_circuits and pchlenski/gpt2-transcoders

    def __init__(self, model, transcoder, layer, cfg: MidcoderConfig):
        super().__init__()
        self.model = model
        self.transcoder = transcoder
        self.layer = layer
        self.cfg = cfg
        self.device = cfg.device
        self.pre_hook_point = model.blocks[layer].ln2.hook_normalized
        self.post_hook_point = model.blocks[layer].hook_mlp_out

        torch.manual_seed(self.cfg.seed)

        # transcoder weights
        self.b_dec = self.transcoder.b_dec.clone().detach()
        self.W_enc = self.transcoder.W_enc.clone().detach()
        self.b_enc = self.transcoder.b_enc.clone().detach()
        self.W_dec = self.transcoder.W_dec.clone().detach()
        self.b_dec_out = self.transcoder.b_dec_out.clone().detach()

        self.W_in = self.model.blocks[self.layer].mlp.W_in.clone().detach()
        self.b_in = self.model.blocks[self.layer].mlp.b_in.clone().detach()
        self.W_out = self.model.blocks[self.layer].mlp.W_out.clone().detach()
        self.b_out = self.model.blocks[self.layer].mlp.b_out.clone().detach()
        
        n_feat = self.W_enc.shape[1]
        d_mlp = self.model.blocks[layer].mlp.W_in.shape[1]
        self.W_mid = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(n_feat, d_mlp, device=self.device)
            ),
            requires_grad=True
        )
        self.b_mid = nn.Parameter(torch.zeros(d_mlp,  device=self.device), requires_grad=True)

    def forward(self, x):
        #x: input into MLP of shape (batch, pos, d_model)
        x = x.to(self.device)
        x = x - self.b_dec
        x = x @ self.W_enc + self.b_enc
        acts = torch.nn.functional.relu(x) #activations
        mid = acts @ self.W_mid + self.b_mid
        out = mid @ self.W_out + self.b_out
        return out, mid, acts
    
    def get_inputs_outputs(self, token_array):
        #x: input to model of shape (batch, pos)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(token_array, stop_at_layer=self.layer+1, 
                                                names_filter=[self.pre_hook_point.name,
                                                              self.post_hook_point.name])
            inputs = cache[self.pre_hook_point.name]
            outputs = cache[self.post_hook_point.name]
        return inputs, outputs
    
    def step(self, batch):
        inputs, outputs = self.get_inputs_outputs(batch)
        out, _, _ = self.forward(inputs)
        loss = nn.MSELoss()(out, outputs)
        scale = (outputs**2).mean()
        loss_norm = loss/scale
        return loss, loss_norm

    def get_dataset(self):
        dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True,
                                trust_remote_code=True)
        dataset = dataset.shuffle(seed=42, buffer_size=10_000)
        tokenized_dataset = tokenize_and_concatenate(dataset, self.model.tokenizer, 
                            max_length=self.cfg.context_length, streaming=True
                            )
        return tokenized_dataset

    def get_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.cfg.batch_size
        dataset = self.get_dataset()
        return iter(DataLoader(dataset, batch_size=batch_size))
    
    def fit(self):        
        optimizer = AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.epochs)
        
        dataloader = self.get_dataloader()
        
        pbar = tqdm(range(self.cfg.epochs))
        history = []
        
        if self.cfg.log: wandb.init(project=self.cfg.wandb_project, config=self.cfg)
        for _ in pbar:
            epoch = []
            for step in range(self.cfg.steps_per_epoch):
                batch = next(dataloader)['tokens']
                loss, loss_norm = self.train().step(batch)
                epoch += [(loss.item(), loss_norm.item())]
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            metrics = {
                "loss": sum([loss for loss, _ in epoch]) / len(epoch),
                "loss_norm": sum([loss_norm for  _, loss_norm in epoch]) / len(epoch),
                "lr": scheduler.get_last_lr()[0]
            }
            if self.cfg.log: wandb.log(metrics)
            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3e}" for k, v in metrics.items()))

            self.clear_cache()
        
        if self.cfg.log: wandb.finish()
        return DataFrame.from_records(history, columns=['loss', 'loss_norm'])

    def evaluate(self, tokens=2e6):
        dataloader = self.get_dataloader()
        steps = tokens//(self.cfg.context_length * self.cfg.batch_size)
        pbar = tqdm(range(steps))
        
        epoch = []
        for _ in pbar:
            with torch.no_grad():
                batch = next(dataloader)['tokens']
                inputs, outputs = self.get_inputs_outputs(batch)
                mid_out, _, acts = self.eval().forward(inputs)
                trans_out = acts @ self.W_dec + self.b_dec_out

                loss, mid_loss, trans_loss = self.get_reconstruction_losses(batch,
                                                                            mid_out,
                                                                            trans_out
                                                                            )
                mid_mse_loss = nn.MSELoss()(mid_out, outputs)
                trans_mse_loss = nn.MSELoss()(trans_out, outputs)

                epoch += [(loss.item(), mid_loss.item(), trans_loss.item(),
                          mid_mse_loss, trans_mse_loss)]
        metrics = {
            "loss": sum([step[0] for step in epoch]) / len(epoch),
            "loss_mid": sum([step[1] for step in epoch]) / len(epoch),
            "loss_trans": sum([step[2] for step in epoch]) / len(epoch),
            "mse_loss_mid": sum([step[3] for step in epoch]) / len(epoch),
            "mse_loss_trans": sum([step[4] for step in epoch]) / len(epoch),
        }
        return metrics

    def get_reconstruction_losses(self, batch, mid_out, trans_out):
        hook_point = self.post_hook_point
        loss = self.model(batch, return_type = "loss").mean()

        def replacement_hook(activations, hook):
            return mid_out
        mid_loss = self.model.run_with_hooks(
            batch,
            return_type="loss",
            fwd_hooks=[(hook_point, partial(replacement_hook))],
        ).mean()
        model.reset_hooks()

        def replacement_hook(activations, hook):
            return trans_out
        trans_loss = self.model.run_with_hooks(
            batch,
            return_type="loss",
            fwd_hooks=[(hook_point, partial(replacement_hook))],
        ).mean()
        model.reset_hooks()

        return loss, mid_loss, trans_loss
        


    def save_weights(self, path):
        weights = {key: midcoder.state_dict()[key] for key in ['W_mid', 'b_mid']}
        torch.save(weights, path)

    def load_weights(self, path):
        #assumes weights are first item in list
        data = torch.load(path)
        self.load_state_dict(data[0])

    def clear_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()


def evaluate_midcoder(midcoder, tokens = 2e6):
    cfg = midcoder.cfg
    dataloader = midcoder.get_dataloader()
    steps = tokens//(cfg.context_length * cfg.batch_size)
    pbar = tqdm(range(steps))
    
    for _ in pbar:
        with torch.no_grad():
            batch = next(dataloader)['tokens']
            loss, loss_norm = self.eval().step(batch)
            
        
        metrics = {
            "loss": sum([loss for loss, _ in epoch]) / len(epoch),
            "loss_norm": sum([loss_norm for  _, loss_norm in epoch]) / len(epoch),
        }
        if self.cfg.log: wandb.log(metrics)
        history.append(metrics)
        pbar.set_description(', '.join(f"{k}: {v:.3f}" for k, v in metrics.items()))

        self.clear_cache()