import torch
import torch.nn as nn
from dataclasses import dataclass
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import wandb
from einops import *
import matplotlib.pyplot as plt
import numpy as np


class ConstrainedAdam(Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    """
    def __init__(self, params, constrained_params, lr, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.constrained_params = list(constrained_params)
    
    def step(self, closure=None):
        with t.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=0, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure)
        with t.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=0, keepdim=True)


@dataclass
class SAEFactorizerConfig():

    seed: int = 42
    lr = 1e-3
    min_lr = 1e-6
    weight_decay = 0.0
    batch_size = 1024
    epochs = 100
    log = False
    wandb_project = 'sae_factorizer'
    device = 'cuda'

    factors = 10000
    topk = 5
    mse_param = 1

    beta1 = 0.9
    beta2 = 0.99


class SAEFactorizer(torch.nn.Module):
    def __init__(self, cfg, inputs):
        super().__init__()
        self.cfg = cfg
        self.inputs = inputs.detach().to(cfg.device)
        self.inputs = self.inputs / self.inputs.norm(dim=-1, keepdim=True) #[n_feat, d_model]

        W_dec = self.get_init_weights()
        self.decoder = nn.Linear(W_dec.shape[1], W_dec.shape[0], bias=False)
        self.decoder.weight = nn.Parameter(W_dec)
        self.encoder = nn.Linear(W_dec.shape[0], W_dec.shape[1], bias=False)
        self.encoder.weight = nn.Parameter(W_dec.T)
        self.b_pre = nn.Parameter(torch.zeros(inputs.shape[-1]))

    def get_init_weights(self, k=5):
        W_dec = self.inputs[torch.randint(self.inputs.shape[0], size=(self.cfg.factors, k))]
        W_dec = W_dec.mean(dim=1)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
        return W_dec.T
    
    def forward(self, x):
        x = x - self.b_pre
        pre_acts = x @ self.W_enc
        topk = torch.topk(pre_acts, k=self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk.indices, topk.values)
        x_hat = acts @ self.W_dec + self.b_pre
        return x_hat, acts

    def step(self, x):
        x_hat, acts = self.forward(x)
        
        metrics = {}
        mse_loss = (x - x_hat)**2
        metrics['nmse_loss'] = (mse_loss.sum(dim=1) /(x**2).sum(dim=1)).mean()

        metrics['L1'] = acts.abs().sum(dim=1).mean()
        metrics['dead_frac'] = (acts.sum(dim=0) == 0).sum()
        
        metrics['loss'] = mse_loss

        self.factor_acts += acts.sum(dim=0)
        return metrics

    def fit(self):
        if self.cfg.log: wandb.init(project=self.cfg.wandb_project, config=self.cfg)

        optimizer = ConstrainedAdam(self.parameters(), self.decoder.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                        betas=(self.cfg.beta1, self.cfg.beta2))
        scheduler = self.get_scheduler(optimizer)
        dataloader = self.get_dataloader()
                
        pbar = tqdm(range(self.cfg.epochs))
        for _ in pbar:
            epoch = []
            self.factor_acts = torch.zeros(self.cfg.factors)

            for batch in dataloader:
                metrics = self.train().step(batch)
                loss = metrics['loss']
                epoch += [{key:val.item() for key, val in metrics.items()}]
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            
            labels = epoch[0].keys()
            metrics = {label: sum([step[label] for step in epoch]) / len(epoch) for label in enumerate(labels)}
            metrics['dead_frac'] = ((self.factor_acts==0).float().sum() / self.cfg.factors).item()

            scheduler.step(metrics['loss'])

            if self.cfg.log: wandb.log(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3e}" for k, v in metrics.items()))
            
        if self.cfg.log: wandb.finish()
        return
    
    def get_dataloader(self):
        return DataLoader(self.decoders, batch_size=self.cfg.batch_size, shuffle=True)

    def get_scheduler(self, optimizer):
        gamma = (self.cfg.min_lr / self.cfg.lr)**(1/self.cfg.epochs)
        return ExponentialLR(optimizer, gamma)

        # return ReduceLROnPlateau(optimizer, threshold=1e-2)