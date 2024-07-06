import torch
import torch.nn as nn
from dataclasses import dataclass
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import wandb
from einops import *


@dataclass
class FactorizerConfig():

    seed: int = 42
    lr = 1e-3
    weight_decay = 0.0
    batch_size = 1000
    epochs = 100
    log = True
    wandb_project = 'factorizer'
    device = 'cuda'

    theta = 0.3
    activation = 'sigmoid'
    factors = 10000
    decoder_param = 1
    factor_param = 1


class Factorizer(torch.nn.Module):
    def __init__(self, cfg, decoders):
        super().__init__()
        self.cfg = cfg
        self.decoders = decoders.detach().to(cfg.device)
        self.decoders = self.decoders / self.decoders.norm(dim=1, keepdim=True) #[n_feat, d_model]

        n_feat, d_model = decoders.shape
        # factors = torch.randn(cfg.factors, d_model).to(cfg.device)
        # factors = self.decoders[:cfg.factors]
        factors = self.decoders[torch.randint(self.decoders.shape[0], size=(self.cfg.factors,))]
        self.factors = torch.nn.Parameter(factors/factors.norm(dim=1,keepdim=True))

    def get_factors(self):
        return self.factors / self.factors.norm(dim=1, keepdim=True)

    def sim_activation(self, x):
        if self.cfg.activation=='tanh':
            return nn.functional.tanh((x/self.cfg.theta)**2)
        elif self.cfg.activation=='sigmoid':
            return nn.functional.sigmoid((x-self.cfg.theta)/0.03) + nn.functional.sigmoid(-(x+self.cfg.theta)/0.03)
        elif self.cfg.activation=='thresh':
            return (x.abs() > self.cfg.theta).float()

    def step(self, batch):
        batch = batch.to(self.cfg.device)
        factors = self.get_factors()
        acts = batch @ factors.T
        acts = self.sim_activation(acts) * acts
        decoder_hat = acts @ factors
        
        mse_loss = (batch - decoder_hat)**2
        mse_loss = (mse_loss.sum(dim=1) /(batch**2).sum(dim=1)).mean()

        decoder_sims_hat = decoder_hat @ decoder_hat.T
        decoder_sims = batch @ batch.T
        weight = self.sim_activation(decoder_sims)
        decoder_sim_loss = weight * (decoder_sims - decoder_sims_hat)**2
        decoder_sim_loss = (decoder_sim_loss.sum(dim=1)/(weight*decoder_sims**2).sum(dim=1)).mean()

        factor_sims = factors @ factors.T
        identity = torch.eye(*factor_sims.shape).to(self.cfg.device)
        weight = self.sim_activation(factor_sims)
        factor_sim_loss = weight * (factor_sims - identity)**2
        factor_sim_loss = (factor_sim_loss.sum(dim=1)/(weight*identity**2).sum(dim=1)).mean()

        loss = mse_loss + self.cfg.decoder_param * decoder_sim_loss + self.cfg.factor_param * factor_sim_loss

        return loss, mse_loss, decoder_sim_loss, factor_sim_loss

    def fit(self):
        if self.cfg.log: wandb.init(project=self.cfg.wandb_project, config=self.cfg)

        optimizer = AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.epochs)
        dataloader = self.get_dataloader()
                
        pbar = tqdm(range(self.cfg.epochs))
        history = []
        for _ in pbar:
            epoch = []
            for batch in dataloader:
                loss, mse_loss, decoder_sim_loss, factor_sim_loss = self.train().step(batch)
                epoch += [(loss.item(), mse_loss.item(), decoder_sim_loss.item(), factor_sim_loss.item())]
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            labels = ['loss', 'mse_loss', 'dec sim loss', 'factor loss']
            metrics = {label: sum([step[i] for step in epoch]) / len(epoch) for i, label in enumerate(labels)}

            if self.cfg.log: wandb.log(metrics)
            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3e}" for k, v in metrics.items()))
            
        if self.cfg.log: wandb.finish()
        return

    def get_dataloader(self):
        return DataLoader(self.decoders, batch_size=self.cfg.batch_size, shuffle=True)