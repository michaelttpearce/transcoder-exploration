import torch
import torch.nn as nn
from dataclasses import dataclass
from datasets import load_dataset
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import wandb
from einops import *
import matplotlib.pyplot as plt
import numpy as np
from utils import tokenize_and_concatenate



class ConstrainedAdam(Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    """
    def __init__(self, params, constrained_params, lr, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.constrained_params = list(constrained_params)
    
    def step(self, closure=None):
        with torch.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=0, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure)
        with torch.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=0, keepdim=True)


@dataclass
class SAEConfig():

    seed: int = 42
    lr = 1e-3
    min_lr = 1e-6
    weight_decay = 0.0
    batch_size = 1024
    epochs = 100
    log = False
    wandb_project = 'meta_sae'
    run_name = None
    device = 'cuda'

    features = 10000
    topk = 5
    mse_param = 1
    aux_mse_param = 0
    feature_sim_param = 0
    feature_thresh = 0.15
    feature_activation = 'tanh'
    

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    dataset = 'Skylion007/openwebtext'
    context_length = 128


class BaseSAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass
    
    def get_inputs(self, x):
        return x

    def feature_sim_activation(self,x):
        if self.cfg.feature_activation=='tanh':
            return nn.functional.tanh((x/self.cfg.feature_thresh)**2)
        elif self.cfg.feature_activation=='sigmoid':
            return nn.functional.sigmoid((x-self.cfg.feature_thresh)/0.02) + nn.functional.sigmoid(-(x+self.cfg.feature_thresh)/0.02)
        elif self.cfg.feature_activation=='threshold':
            return (x.abs() > self.cfg.feature_thresh).float()

    def step(self, batch):
        x = self.get_inputs(batch)
        x_hat, acts, err_hat = self.forward(x)

        dims = tuple(i for i in range(len(acts.shape)-1)) #all but last dim
        self.feature_acts += acts.sum(dim=dims)
        
        metrics = {}
        mse_loss = (x - x_hat)**2
        metrics['mse_loss'] = mse_loss.mean()
        metrics['nmse_loss'] = (mse_loss.sum(dim=-1) / (x**2).sum(dim=-1)).mean()

        if self.cfg.aux_mse_param > 0:
            err = (x - x_hat).detach()
            aux_mse_loss = (err - err_hat)**2
            metrics['aux_mse_loss'] = aux_mse_loss.mean()
            metrics['aux_nmse_loss'] = (aux_mse_loss.sum(dim=-1) / (x**2).sum(dim=-1)).mean()
        else:
            metrics['aux_mse_loss'] = torch.tensor(0)

        if self.cfg.feature_sim_param > 0:
            features = self.decoder.weight.T
            active = (acts > 0).sum(dim=dims) > 0
            feature_sims = features[active] @ features[active].T
            identity = torch.eye(*feature_sims.shape).to(self.cfg.device)
            weight = self.feature_sim_activation(feature_sims)
            feature_sim_loss =  weight * (feature_sims - identity)**2
            feature_sim_loss = (feature_sim_loss.sum(dim=-1)).mean()
            metrics['feature_sim_loss'] = feature_sim_loss
        else:
            feature_sim_loss = torch.tensor(0)

        metrics['L1'] = acts.abs().sum(dim=-1).mean()

        metrics['L0'] = (acts > 0).float().sum(dim=-1).mean()
        
        metrics['loss'] = self.cfg.mse_param * metrics['mse_loss'] + self.cfg.aux_mse_param * metrics['aux_mse_loss'] + self.cfg.feature_sim_param * feature_sim_loss

        return metrics

    def fit(self):
        if self.cfg.log and (self.cfg.run_name is not None): 
            wandb.init(project=self.cfg.wandb_project, 
                                    config=self.cfg,
                                    name = self.cfg.run_name)
        elif self.cfg.log:
            wandb.init(project=self.cfg.wandb_project, 
                                    config=self.cfg)

        if self.decoder.weight.requires_grad:
            optimizer = ConstrainedAdam(self.parameters(), self.decoder.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                        betas=(self.cfg.beta1, self.cfg.beta2), eps = self.cfg.eps)
        else:
            optimizer = Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                        betas=(self.cfg.beta1, self.cfg.beta2), eps = self.cfg.eps)
        
        scheduler = self.get_scheduler(optimizer)
        dataloader = self.get_dataloader()
                
        num_epochs, steps_per_epoch = self.get_epochs(dataloader)
        pbar = tqdm(range(num_epochs))
        for _ in pbar:
            epoch = []
            self.feature_acts = torch.zeros(self.cfg.features).to(self.cfg.device) #used to identify dead features at epoch level

            for step, batch in enumerate(dataloader):
                metrics = self.train().step(batch)
                loss = metrics['loss']
                epoch += [{key:val.item() for key, val in metrics.items()}]
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (steps_per_epoch is not None) and (step >= steps_per_epoch-1):
                    break
            
            labels = epoch[0].keys()
            metrics = {label: sum([step[label] for step in epoch]) / len(epoch) for label in labels}
            metrics['dead_frac'] = ((self.feature_acts==0).float().sum() / self.cfg.features).item()

            scheduler.step(metrics['loss'])

            if self.cfg.log: wandb.log(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3e}" for k, v in metrics.items()))
            
        if self.cfg.log: wandb.finish()
        return
    
    def get_dataloader(self):
        pass
    
    def get_epochs(self, dataloader):
        return self.cfg.epochs, None

    def get_scheduler(self, optimizer):
        gamma = (self.cfg.min_lr / self.cfg.lr)**(1/self.cfg.epochs)
        return ExponentialLR(optimizer, gamma)



class BaseTopKSAE(BaseSAE):
    
    def forward(self, x):
        x = x - self.b_pre
        pre_acts = self.encoder(x)
        topk = torch.topk(pre_acts, k=self.cfg.topk, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk.indices, topk.values)
        x_hat = self.decoder(acts) + self.b_pre
        
        if self.cfg.aux_mse_param > 0:
            # dead_over_batch = ((acts.sum(dim=-1)) == 0).float()
            # aux_acts = acts * dead_over_batch.unsqueeze(-1)
            # err_hat = self.decoder(aux_acts)

            aux_topk = torch.topk(pre_acts, k=self.cfg.topk + self.cfg.aux_topk, dim=-1)
            aux_acts = torch.zeros_like(pre_acts)

            slice_idxs = torch.arange(self.cfg.topk, self.cfg.topk+self.cfg.aux_topk).to(self.cfg.device)
            idxs = aux_topk.indices.index_select(-1, slice_idxs)
            vals = aux_topk.values.index_select(-1, slice_idxs)
            aux_acts.scatter_(-1, idxs, vals)
            err_hat = self.decoder(aux_acts)
        else:
            err_hat = None
        return x_hat, acts, err_hat
  

class MetaSAE(BaseTopKSAE):
    def __init__(self, cfg, inputs):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.inputs = inputs.detach().to(cfg.device)
        self.inputs = self.inputs / self.inputs.norm(dim=-1, keepdim=True) #[n_feat, d_model]

        W_dec = self.get_init_weights().to(self.cfg.device)
        self.decoder = nn.Linear(W_dec.shape[1], W_dec.shape[0], bias=False)
        self.decoder.weight = nn.Parameter(W_dec)
        self.encoder = nn.Linear(W_dec.shape[0], W_dec.shape[1], bias=False)
        self.encoder.weight = nn.Parameter(W_dec.T)
        self.b_pre = nn.Parameter(torch.zeros(inputs.shape[-1]).to(self.cfg.device))
    
    def get_init_weights(self):
        # W_dec = self.inputs[torch.randint(self.inputs.shape[0], size=(self.cfg.features, 20))]  
        # W_dec = W_dec.mean(dim=1)  #[meta_features, d_model]
        # W_dec += W_dec.std() * torch.randn(*W_dec.shape).to(W_dec.device)
        # W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)

        W_dec = torch.randn(self.cfg.features, self.inputs.shape[1]).to(self.cfg.device)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
        return W_dec.T #[d_model, meta_features]
    
    def get_dataloader(self):
        dataloader = DataLoader(self.inputs, batch_size=self.cfg.batch_size, shuffle=True)
        return dataloader

    def evaluate(self):
        with torch.no_grad():
            features = self.decoder.weight.T.detach()
            x_hat, acts = self.forward(self.inputs)

        #plot of num features per decoder direction
        features_per_decoder = (acts > 0).float().sum(dim=1)

        plt.figure(figsize=(5,4), dpi=150)
        plt.hist(features_per_decoder.cpu(), 15)
        plt.yscale('log')
        plt.xlabel('Active meta-features per feature')
        plt.ylabel('Counts')
        plt.title('Meta-features per feature')

        #plot of num decoders per feature
        decoders_per_feature = (acts > 0).float().sum(dim=0)

        plt.figure(figsize=(5,4), dpi=150)
        plt.hist(decoders_per_feature.cpu(), 30)
        plt.yscale('log')
        plt.xlabel('Features per meta-feature')
        plt.ylabel('Counts')
        plt.title('Features per meta-feature')

        #plot of cos sims distribution
        cos_sims = nn.functional.cosine_similarity(self.inputs, x_hat, dim=1)

        plt.figure(figsize=(5,4), dpi=150)
        plt.hist(cos_sims.cpu(), 100)
        plt.yscale('log')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Counts')
        plt.title('Feature vs Reconstruction')

        #plot of err norm
        err_norm = (self.inputs - x_hat).norm(dim=1)

        plt.figure(figsize=(5,4), dpi=150)
        plt.hist(err_norm.cpu(), 100)
        plt.yscale('log')
        plt.xlabel('Error Norm')
        plt.ylabel('Counts')
        plt.title('Feature vs Reconstruction')


class UberSAE(BaseTopKSAE):
    
    def __init__(self, cfg, model_dict, model_cfg):
        super().__init__()
        # model_dict: {'model': HookedTransformer}
        # model_cfg: dict of {'sae': sae, 'meta_sae': MetaSAE, 'hook_point': hook_point, 'layer': layer} to avoid parameters being added
        self.cfg = cfg
        self.device = cfg.device
        self.model_dict = model_dict
        self.model_cfg = model_cfg

        torch.manual_seed(self.cfg.seed)

        W_dec, W_enc = self.get_init_weights()
        self.decoder = nn.Linear(W_dec.shape[0], W_dec.shape[1], bias=False)
        self.decoder.weight = nn.Parameter(W_dec.T, requires_grad=False)
        
        self.encoder = nn.Linear(W_enc.shape[0], W_enc.shape[1], bias=False)
        self.encoder.weight = nn.Parameter(W_enc.T)
        # self.encoder.weight = nn.Parameter(torch.randn_like(W_enc.T).to(self.device))

        self.b_pre = nn.Parameter(torch.zeros(W_enc.shape[0]).to(self.cfg.device))
    
    def get_init_weights(self):
        W_dec = self.model_cfg['W_dec']  #[uber_features, d_model]
        W_enc = torch.linalg.pinv(W_dec) #[d_model, uber_features]
        self.cfg.features = W_dec.shape[0]
        return W_dec.to(self.cfg.device), W_enc.to(self.cfg.device) #[uber_features, d_model], [d_model, uber_features]
    
    def get_inputs(self, batch):
        #x: input to model of shape (batch, pos)
        tokens = batch['tokens']
        with torch.no_grad():
            _, cache = self.model_dict['model'].run_with_cache(tokens, stop_at_layer=self.model_cfg['layer']+1, 
                                                names_filter=[self.model_cfg['hook_name']])
            inputs = cache[self.model_cfg['hook_name']]
        return inputs
    
    def get_epochs(self, dataloader):
        num_epochs = int(self.cfg.train_tokens/(self.cfg.context_length * self.cfg.batch_size * self.cfg.steps_per_epoch)) + 1
        return num_epochs, self.cfg.steps_per_epoch
    
    def get_dataset(self):
        dataset = load_dataset(self.cfg.dataset, split='train', streaming=True,
                                trust_remote_code=True)
        dataset = dataset.shuffle(seed=self.cfg.seed, buffer_size=10_000)
        tokenized_dataset = tokenize_and_concatenate(dataset, self.model_dict['model'].tokenizer, 
                            max_length=self.cfg.context_length, streaming=True
                            )
        return tokenized_dataset

    def get_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.cfg.batch_size
        dataset = self.get_dataset()
        return iter(DataLoader(dataset, batch_size=batch_size))


def define_uber_sae_model_cfg(model, sae, meta_sae, device='cuda'):
    model_dict = {'model':model}
    model_cfg = {'model_cfg':model.cfg, 
                 'sae_cfg': sae.cfg,
                 'meta_cfg': meta_sae.cfg,
                 'hook_name': sae.cfg.hook_name,
                 'layer':sae.cfg.hook_layer,
                 'sae_size': sae.cfg.d_sae,
                 'meta_size': meta_sae.cfg.features}

    sae_dec = sae.W_dec.detach().to(device)
    sae_dec = sae_dec / sae_dec.norm(dim=1, keepdim=True)
    with torch.no_grad():
        sae_dec_hat, _, _ = meta_sae.forward(sae_dec)
        sae_dec_error = sae_dec - sae_dec_hat
        sae_dec_error = sae_dec_error / sae_dec_error.norm(dim=1, keepdim=True)
    meta_dec = meta_sae.decoder.weight.T.detach().to(device) #[meta_features, d_model]

    W_dec = torch.cat([meta_dec, sae_dec_error], dim=0)
    model_cfg['W_dec'] = W_dec / W_dec.norm(dim=1, keepdim=True)  #[uber_features, d_model]
    return model_dict, model_cfg




