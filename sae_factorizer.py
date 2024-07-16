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
class FactorizerConfig():

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
    factor_thresh = 0.15
    factor_activation = 'tanh'
    factor_sim_param = 1

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8


class BaseFactorizer(torch.nn.Module):
    def __init__(self, cfg, inputs):
        super().__init__()
        self.cfg = cfg
        self.inputs = inputs.detach().to(cfg.device)
        self.inputs = self.inputs / self.inputs.norm(dim=-1, keepdim=True) #[n_feat, d_model]

        W_dec = self.get_init_weights().to(self.cfg.device)
        self.decoder = nn.Linear(W_dec.shape[1], W_dec.shape[0], bias=False)
        self.decoder.weight = nn.Parameter(W_dec)
        self.encoder = nn.Linear(W_dec.shape[0], W_dec.shape[1], bias=False)
        self.encoder.weight = nn.Parameter(W_dec.T)
        self.b_pre = nn.Parameter(torch.zeros(inputs.shape[-1]).to(self.cfg.device))
    
    def get_init_weights(self):
        W_dec = self.inputs[torch.randint(self.inputs.shape[0], size=(self.cfg.factors, 20))]
        W_dec = W_dec.mean(dim=1)
        W_dec += W_dec.std() * torch.randn(*W_dec.shape).to(W_dec.device)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
        return W_dec.T
    
    def forward(self, x):
        pass

    def factor_sim_activation(self,x):
        if self.cfg.factor_activation=='tanh':
            return nn.functional.tanh((x/self.cfg.factor_thresh)**2)
        elif self.cfg.factor_activation=='sigmoid':
            return nn.functional.sigmoid((x-self.cfg.factor_thresh)/0.02) + nn.functional.sigmoid(-(x+self.cfg.theta)/0.02)
        elif self.cfg.factor_activation=='threshold':
            return (x.abs() > self.cfg.factor_thresh).float()

    def step(self, x):
        x_hat, acts = self.forward(x)
        
        metrics = {}
        mse_loss = (x - x_hat)**2
        metrics['nmse_loss'] = (mse_loss.sum(dim=1) / (x**2).sum(dim=1)).mean()

        if self.cfg.factor_sim_param > 0:
            factors = self.decoder.weight.T
            active = (acts > 0).sum(dim=0) > 0
            factor_sims = factors[active] @ factors[active].T
            identity = torch.eye(*factor_sims.shape).to(self.cfg.device)
            weight = self.factor_sim_activation(factor_sims)
            factor_sim_loss =  weight * (factor_sims - identity)**2
            factor_sim_loss = (factor_sim_loss.sum(dim=1)).mean()
            metrics['factor_sim_loss'] = factor_sim_loss
        else:
            factor_sim_loss = 0

        metrics['L1'] = acts.abs().sum(dim=1).mean()

        metrics['L0'] = (acts > 0).float().sum(dim=1).mean()
        
        metrics['loss'] = self.cfg.mse_param * mse_loss.sum(dim=1).mean() + self.cfg.factor_sim_param * factor_sim_loss

        self.factor_acts += acts.sum(dim=0)
        return metrics

    def fit(self):
        if self.cfg.log: wandb.init(project=self.cfg.wandb_project, config=self.cfg)

        optimizer = ConstrainedAdam(self.parameters(), self.decoder.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                        betas=(self.cfg.beta1, self.cfg.beta2), eps = self.cfg.eps)
        scheduler = self.get_scheduler(optimizer)
        dataloader = self.get_dataloader()
                
        pbar = tqdm(range(self.cfg.epochs))
        for _ in pbar:
            epoch = []
            self.factor_acts = torch.zeros(self.cfg.factors).to(self.cfg.device)

            for batch in dataloader:
                metrics = self.train().step(batch)
                loss = metrics['loss']
                epoch += [{key:val.item() for key, val in metrics.items()}]
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            
            labels = epoch[0].keys()
            metrics = {label: sum([step[label] for step in epoch]) / len(epoch) for label in labels}
            metrics['dead_frac'] = ((self.factor_acts==0).float().sum() / self.cfg.factors).item()

            scheduler.step(metrics['loss'])

            if self.cfg.log: wandb.log(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3e}" for k, v in metrics.items()))
            
        if self.cfg.log: wandb.finish()
        return
    
    def get_dataloader(self):
        return DataLoader(self.inputs, batch_size=self.cfg.batch_size, shuffle=True)

    def get_scheduler(self, optimizer):
        gamma = (self.cfg.min_lr / self.cfg.lr)**(1/self.cfg.epochs)
        return ExponentialLR(optimizer, gamma)
    
    def evaluate(self):
        with torch.no_grad():
            factors = self.decoder.weight.T.detach()
            x_hat, acts = self.forward(self.inputs)

        #plot of num factors per decoder direction
        factors_per_decoder = (acts > 0).float().sum(dim=1)

        plt.figure(figsize=(5,4), dpi=150)
        plt.hist(factors_per_decoder.cpu(), 15)
        plt.yscale('log')
        plt.xlabel('Active factors per decoder')
        plt.ylabel('Counts')
        plt.title('Factors per decoder')

        #plot of num decoders per factor
        decoders_per_factor = (acts > 0).float().sum(dim=0)

        plt.figure(figsize=(5,4), dpi=150)
        plt.hist(decoders_per_factor.cpu(), 30)
        plt.yscale('log')
        plt.xlabel('Decoders per factor')
        plt.ylabel('Counts')
        plt.title('Decoders per factor')

        #plot of cos sims distribution
        cos_sims = nn.functional.cosine_similarity(self.inputs, x_hat, dim=1)

        plt.figure(figsize=(5,4), dpi=150)
        plt.hist(cos_sims.cpu(), 100)
        plt.yscale('log')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Counts')
        plt.title('Decoder vs Reconstruction')

        #plot of err norm
        err_norm = (self.inputs - x_hat).norm(dim=1)

        plt.figure(figsize=(5,4), dpi=150)
        plt.hist(err_norm.cpu(), 100)
        plt.yscale('log')
        plt.xlabel('Error Norm')
        plt.ylabel('Counts')
        plt.title('Decoder vs Reconstruction')



class TopKFactorizer(BaseFactorizer):
    
    def forward(self, x):
        x = x - self.b_pre
        pre_acts = self.encoder(x)
        topk = torch.topk(pre_acts, k=self.cfg.topk, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk.indices, topk.values)
        x_hat = self.decoder(acts) + self.b_pre
        return x_hat, acts


class ThresholdFactorizer(BaseFactorizer):
    
    def activation(self, x):
        if self.cfg.activation=='tanh':
            return nn.functional.tanh((x/self.cfg.theta)**2)
        elif self.cfg.activation=='sigmoid':
            return nn.functional.sigmoid((x-self.cfg.theta)/0.03) + nn.functional.sigmoid(-(x+self.cfg.theta)/0.03)
        elif self.cfg.activation=='threshold':
            return (x.abs() > self.cfg.theta).float()
        
    def forward(self, x):
        x = x - self.b_pre
        pre_acts = self.encoder(x)
        # pre_acts = x @ self.decoder.weight
        acts = self.activation(pre_acts) * pre_acts
        x_hat = self.decoder(acts) + self.b_pre
        return x_hat, acts