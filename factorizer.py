import torch
import torch.nn as nn
from dataclasses import dataclass
from tqdm import tqdm
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import wandb
from einops import *
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FactorizerConfig():

    seed: int = 42
    lr = 1e-3
    min_lr = 1e-6
    weight_decay = 0.0
    batch_size = 1000
    epochs = 100
    log = True
    wandb_project = 'factorizer'
    device = 'cuda'

    theta = 0.2
    activation = 'threshold'
    factors = 10000
    decoder_param = 1
    factor_param = 1
    l1_param = 0.1

    beta1 = 0.9
    beta2 = 0.95


class Factorizer(torch.nn.Module):
    def __init__(self, cfg, decoders):
        super().__init__()
        self.cfg = cfg
        self.decoders = decoders.detach().to(cfg.device)
        self.decoders = self.decoders / self.decoders.norm(dim=1, keepdim=True) #[n_feat, d_model]

        n_feat, d_model = decoders.shape
        # factors = torch.randn(cfg.factors, d_model).to(cfg.device)
        # factors = self.decoders[:cfg.factors]
        factors = self.get_init_factors()
        self.factors = torch.nn.Parameter(factors/factors.norm(dim=1,keepdim=True))

    def get_init_factors(self):
        dec1 = self.decoders[torch.randint(self.decoders.shape[0], size=(self.cfg.factors,))]
        dec2 = self.decoders[torch.randint(self.decoders.shape[0], size=(self.cfg.factors,))]
        return dec1 + dec2

    def get_factors(self):
        return self.factors / self.factors.norm(dim=1, keepdim=True)

    def sim_activation(self, x):
        if self.cfg.activation=='tanh':
            return nn.functional.tanh((x/self.cfg.theta)**2)
        elif self.cfg.activation=='sigmoid':
            return nn.functional.sigmoid((x-self.cfg.theta)/0.03) + nn.functional.sigmoid(-(x+self.cfg.theta)/0.03)
        elif self.cfg.activation=='threshold':
            return (x.abs() > self.cfg.theta).float()

    def step(self, batch):
        batch = batch.to(self.cfg.device)
        factors = self.get_factors()
        acts_orig = batch @ factors.T
        acts = self.sim_activation(acts_orig) * acts_orig
        decoder_hat = acts @ factors
        
        mse_loss = (batch - decoder_hat)**2
        mse_loss = (mse_loss.sum(dim=1) /(batch**2).sum(dim=1)).mean()

        decoder_sims_hat = decoder_hat @ decoder_hat.T
        decoder_sims = batch @ batch.T
        weight = self.sim_activation(decoder_sims)
        decoder_sim_loss = weight * (decoder_sims - decoder_sims_hat)**2
        decoder_sim_loss = (decoder_sim_loss.sum(dim=1)/(weight*decoder_sims**2).sum(dim=1)).mean()

        factor_activity = (acts_orig > self.cfg.theta).sum(dim=0)
        mask = factor_activity > 0
        factor_sims = factors[mask] @ factors[mask].T
        identity = torch.eye(*factor_sims.shape).to(self.cfg.device)
        # weight = self.sim_activation(factor_sims)
        factor_sim_loss =  (factor_sims - identity)**2
        factor_sim_loss = (factor_sim_loss.sum(dim=1)).mean()

        L1_loss = acts.abs().sum(dim=1).mean()
        
        loss = mse_loss + self.cfg.decoder_param * decoder_sim_loss + self.cfg.factor_param * factor_sim_loss + self.l1_param * L1_loss

        return loss, mse_loss, decoder_sim_loss, factor_sim_loss, L1_loss

    def fit(self):
        if self.cfg.log: wandb.init(project=self.cfg.wandb_project, config=self.cfg)

        optimizer = AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
                        betas=(self.cfg.beta1, self.cfg.beta2))
        scheduler = self.get_scheduler(optimizer)
        dataloader = self.get_dataloader()
                
        pbar = tqdm(range(self.cfg.epochs))
        history = []
        for _ in pbar:
            epoch = []
            for batch in dataloader:
                loss, mse_loss, decoder_sim_loss, factor_sim_loss, L1_loss = self.train().step(batch)
                epoch += [(loss.item(), mse_loss.item(), decoder_sim_loss.item(), factor_sim_loss.item(), L1_loss.item())]
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            
            labels = ['loss', 'mse_loss', 'dec sim loss', 'factor loss', 'L1 loss']
            metrics = {label: sum([step[i] for step in epoch]) / len(epoch) for i, label in enumerate(labels)}

            scheduler.step(metrics['loss'])
            # metrics['lr':scheduler.get_last_lr()[0]]

            if self.cfg.log: wandb.log(metrics)
            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3e}" for k, v in metrics.items()))
            
        if self.cfg.log: wandb.finish()
        return

    def evaluate(self):
        factors = self.get_factors().detach()
        acts_orig = self.decoders @ factors.T
        factors_on = (acts_orig > self.cfg.theta).float()

        decoder_hat = (factors_on * acts_orig) @ factors

        #plot of num factors per decoder direction
        factors_per_decoder = factors_on.sum(dim=1)

        plt.figure(figsize=(5,4), dpi=150)
        plt.hist(factors_per_decoder.cpu(), 15)
        plt.yscale('log')
        plt.xlabel('Active factors per decoder')
        plt.ylabel('Counts')
        plt.title('Factors per decoder')

        #plot of num decoders per factor
        decoders_per_factor = factors_on.sum(dim=0)

        plt.figure(figsize=(5,4), dpi=150)
        plt.hist(decoders_per_factor.cpu(), 30)
        plt.yscale('log')
        plt.xlabel('Decoders per factor')
        plt.ylabel('Counts')
        plt.title('Decoders per factor')

        #plot of cos sims distribution
        cos_sims = nn.functional.cosine_similarity(self.decoders, decoder_hat, dim=1)

        plt.figure(figsize=(5,4), dpi=150)
        plt.hist(cos_sims.cpu(), 100)
        plt.yscale('log')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Counts')
        plt.title('Decoder vs Reconstruction')

        #plot of err norm
        err_norm = (self.decoders - decoder_hat).norm(dim=1)

        plt.figure(figsize=(5,4), dpi=150)
        plt.hist(err_norm.cpu(), 100)
        plt.yscale('log')
        plt.xlabel('Error Norm')
        plt.ylabel('Counts')
        plt.title('Decoder vs Reconstruction')
        
    def get_dataloader(self):
        return DataLoader(self.decoders, batch_size=self.cfg.batch_size, shuffle=True)

    def get_scheduler(self, optimizer):
        gamma = (self.cfg.min_lr / self.cfg.lr)**(1/self.cfg.epochs)
        return ExponentialLR(optimizer, gamma)

        # return ReduceLROnPlateau(optimizer, threshold=1e-2)

def plot_decoder_reconstruction(factorizer, i):
    factors = factorizer.get_factors().detach()
    acts = factorizer.decoders @ factors.T
    acts = factorizer.sim_activation(acts) * acts

    plt.figure(figsize=(9,4),dpi=150)
    plt.subplot(1,2,1)
    plt.hist(acts[i].cpu(),100)
    plt.yscale('log')
    plt.xlabel('Factor Cos Sim')
    plt.ylabel('Count')
    plt.title(f'Decoder {i}')

    plt.subplot(1,2,2)
    y = acts[i] @ factors
    x = factorizer.decoders[i]
    plt.plot(x.cpu(), y.cpu(), '.', alpha=0.3)
    vec = np.arange(-0.15, 0.15, 0.01)
    plt.plot(vec, vec, 'k--')
    plt.xlabel('Decoder')
    plt.ylabel('Decoder Reconstruction')
    plt.title(f'Err norm: {(y-x).norm():.2f}, Cos Sim: {torch.nn.functional.cosine_similarity(x,y,dim=0):.2f}')
    plt.tight_layout()

    feat_ids = torch.arange(acts.shape[1]).cuda()[acts[i].abs() > 0]
    print(f'Factor ids: {feat_ids}')
    print(f'Factor activations: {acts[i,feat_ids]}')

    x = factors[feat_ids]
    print(f'Factor cosine sims')
    print(x @ x.T)

def plot_factor_activations(factorizer,i):
    factors = factorizer.get_factors().detach()
    acts = factorizer.decoders @ factors.T
    acts = factorizer.sim_activation(acts) * acts
    
    plt.figure(figsize=(5,4),dpi=150)
    plt.hist(acts[:,i].cpu(),100)
    plt.yscale('log')
    plt.xlabel('Activations')
    plt.ylabel('Count')
    plt.title(f'Decoders for Factor {i}')

    dec_ids = torch.arange(acts.shape[0]).cuda()[acts[:,i].abs() > 0]
    print(dec_ids)