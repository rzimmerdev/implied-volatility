import os

import lightning
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from . import SSV
from ..datasets import ParamDataset, SSVDataset
from .transformer import TransformerEncoder
from ..sabr import ParametricSABR


class SST(nn.Module):
    def __init__(self,
                 n,
                 heads,
                 num_layers,
                 out_features=None):
        nn.Module.__init__(self)
        self.n = n
        self.transformer = TransformerEncoder(n, heads, num_layers, 4, 0.5, out_features)

    @classmethod
    def best_candidates(cls, values, scores, p=0.4):
        # use values to return only the top p% of the inputs
        n = len(values)
        top = int((1 - p) * n)
        indices = torch.argsort(scores, dim=0, descending=True)
        selected = indices[top:]
        return values[selected]

    def forward(self, x):
        return self.transformer(x)


class LitSST(lightning.LightningModule):
    def __init__(self, in_features, heads, num_layers, out_features=None, checkpoint_path=None, lr=1e-3):
        super(LitSST, self).__init__()
        self.model = SST(in_features, heads, num_layers, out_features)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.losses = []

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

    def forward(self, x):
        return self.model(x)[:, :, 0]

    def training_step(self, batch, batch_idx):
        x = torch.clamp(batch[0], -1e9, 1e9)
        y = batch[1]
        output = self(x)

        loss = self.criterion(output, y)

        if loss is None or np.isnan(loss.item()) or np.isinf(loss.item()):
            raise ValueError(f"Loss is None for batch {batch_idx}")

        # log loss
        self.losses.append(loss.item())

        return loss

    def configure_optimizers(self):
        return self.optimizer

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)
        print(f'Model saved to {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))
        print(f'Model loaded from {checkpoint_path}')


class MultiSST:
    def __init__(self, in_features, heads, num_layers, out_features=None):
        self.z_alpha = LitSST(in_features, heads, num_layers, out_features)
        self.z_rho = LitSST(in_features, heads, num_layers, out_features)
        self.z_volvol = LitSST(in_features, heads, num_layers, out_features)

    def exists(self, checkpoint_path):
        return all([os.path.exists(f"{checkpoint_path}/{name}.pth") for name in ("alpha", "rho", "volvol")])

    def load_checkpoint(self, checkpoint_path):
        for name, model in zip(("alpha", "rho", "volvol"), (self.z_alpha, self.z_rho, self.z_volvol)):
            model.load_checkpoint(f"{checkpoint_path}/{name}.pth")
        return self

    def save_checkpoint(self, checkpoint_path):
        for name, model in zip(("alpha", "rho", "volvol"), (self.z_alpha, self.z_rho, self.z_volvol)):
            model.save_checkpoint(f"{checkpoint_path}/{name}.pth")

    def train(self, ssv_dataset, epochs=100):
        data_p = ParamDataset(ssv_dataset, "alpha")
        data_q = ParamDataset(ssv_dataset, "rho")
        data_r = ParamDataset(ssv_dataset, "volvol")

        trainer = lightning.Trainer(max_epochs=epochs)
        dataloader_p = torch.utils.data.DataLoader(data_p, batch_size=1, shuffle=True)
        dataloader_q = torch.utils.data.DataLoader(data_q, batch_size=1, shuffle=True)
        dataloader_r = torch.utils.data.DataLoader(data_r, batch_size=1, shuffle=True)

        # train alpha, then rho, then volvol
        trainer.fit(self.z_alpha, dataloader_p)
        trainer.fit(self.z_rho, dataloader_q)
        trainer.fit(self.z_volvol, dataloader_r)

        return self.z_alpha, self.z_rho, self.z_volvol

    def funcs(self):
        return {
            "alpha": self.z_alpha,
            "rho": self.z_rho,
            "volvol": self.z_volvol
        }

    def optim_candidates(self, rows, score_func):
        candidates = rows.iloc[:, 1:3].values

        inputs = torch.tensor(rows.iloc[:, 1:5].values[np.newaxis, :, :], dtype=torch.float32)
        scores = score_func(inputs)[0]  # get only first batch

        return SST.best_candidates(candidates, scores, p=0.4)

    def fit_params(self, inputs):
        score_funcs = self.funcs()

        return ParametricSABR.fit_params({
            key: self.optim_candidates(inputs[key], score_funcs[key]) for key in ("alpha", "rho", "volvol")
        })
