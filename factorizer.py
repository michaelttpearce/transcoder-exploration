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
    factor_param = 0.5


class Factorizer(torch.nn.Module):
    def __init__(self, cfg, decoders):
        super().__init__()
        self.cfg = cfg
        self.decoders = decoders.detach().to(cfg.device)
        self.decoders = self.decoders / self.decoders.norm(dim=1, keepdim=True) #[n_feat, d_model]

        n_feat, d_model = decoders.shape
        factors = torch.randn(cfg.factors, d_model).to(cfg.device)
        self.factors = torch.nn.Parameter(factors/factors.norm(dim=1,keepdim=True))

    def get_factors(self):
        return self.factors / self.factors.norm(dim=1, keepdim=True)

    def sim_weight(self, x):
        if cfg.activation=='tanh':
            return nn.functional.tanh((x/self.cfg.theta)**2)
        elif cfg.activation=='sigmoid':
            return nn.functional.sigmoid((x-self.cfg.theta)/0.03) + nn.functional.sigmoid(-(x+self.cfg.theta)/0.03)

    def step(self, batch):
        batch = batch.to(self.cfg.device)
        factors = self.get_factors()
        d_hat = (batch @ factors.T) @ factors
        
        decoder_sims_hat = d_hat @ d_hat.T
        decoder_sims = batch @ batch.T
        weight = torch.nn.functional.tanh((decoder_sims/self.cfg.theta)**2)
        decoder_sim_loss = torch.mean(weight * (decoder_sims - decoder_sims_hat)**2)

        factor_sims = factors @ factors.T
        identity = torch.eye(*factor_sims.shape).to(self.cfg.device)
        weight = torch.nn.functional.tanh((factor_sims/self.cfg.theta)**2)
        factor_sim_loss = torch.mean(weight * (factor_sims - identity)**2)

        loss = decoder_sim_loss + self.cfg.factor_param * factor_sim_loss

        return loss, decoder_sim_loss, factor_sim_loss

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
                loss, decoder_sim_loss, factor_sim_loss = self.train().step(batch)
                epoch += [(loss.item(), decoder_sim_loss.item(), factor_sim_loss.item())]
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            labels = ['loss', 'dec sim loss', 'factor loss']
            metrics = {label: sum([step[i] for step in epoch]) / len(epoch) for i, label in enumerate(labels)}

            if self.cfg.log: wandb.log(metrics)
            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3e}" for k, v in metrics.items()))
            
        if self.cfg.log: wandb.finish()
        return

    def get_dataloader(self):
        return DataLoader(self.decoders, batch_size=self.cfg.batch_size, shuffle=True)