import os

import lightning
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn
from torch.utils.data import Dataset

from .transformer import TransformerEncoder
from ..datasets import ScoresDataset
from ..sabr import ParametricSABR


class SST(nn.Module):
    def __init__(self, n, *args, **kwargs):
        nn.Module.__init__(self)
        self.n = n
        self.transformer = TransformerEncoder(n, *args, **kwargs)

    @classmethod
    def best_candidates(cls, values, scores, p=0.4):
        # use values to return only the top p% of the inputs
        n = len(values)
        top = int((1 - p) * n)
        indices = torch.argsort(scores, dim=0, descending=False)
        selected = indices[top:]
        return values[selected]

    def forward(self, x):
        return self.transformer(x)


class LitSST(lightning.LightningModule):
    def __init__(self,
                 checkpoint_path=None,
                 lr=1e-3,
                 *args, **kwargs):
        """Initializes a LitSST model with a transformer encoder"""
        super(LitSST, self).__init__()
        self.model = SST(*args, **kwargs)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.losses = []

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

    def forward(self, x):
        return self.model(x)[:, :, 0]

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)

        loss = self.criterion(output, y)
        r_squared = 1 - loss / torch.var(y)

        if loss is None or np.isnan(loss.item()) or np.isinf(loss.item()):
            raise ValueError(f"Loss is None for batch {batch_idx}")

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('r_squared', r_squared, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.losses.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        x = torch.clamp(batch[0], -1e9, 1e9)
        y = batch[1]
        output = self(x)

        loss = self.criterion(output, y)
        r_squared = 1 - loss / torch.var(y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('r_squared', r_squared, on_epoch=True, prog_bar=True, logger=True)

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
    def __init__(self,
                 in_features,
                 heads,
                 num_blocks,
                 num_layers,
                 forward_expansion=4,
                 out_features=None,
                 dropout=0.2,
                 checkpoint_path=None,
                 lr=1e-3):
        """
        Initializes a MultiSST model with three LitSST models for alpha, rho, and volvol

        Args:
            in_features: int, number of input features
            heads: int, number of heads in the self-attention layer
            num_blocks: int, number of transformer blocks
            num_layers: int, number of layers in the feed forward network
            forward_expansion: int, expansion factor for the feed forward network
            out_features: int, number of output features
            checkpoint_path: str, path to load model checkpoint
            lr: float, learning rate for the optimizer
        """
        self.z_alpha, self.z_rho, self.z_volvol = [
            LitSST(checkpoint_path, lr, in_features, heads, num_blocks, num_layers, forward_expansion, out_features,
                   dropout)
            for _ in range(3)
        ]

    def exists(self, checkpoint_path):
        return all([os.path.exists(f"{checkpoint_path}/{name}.pth") for name in ("alpha", "rho", "volvol")])

    def load_checkpoint(self, checkpoint_path):
        versions = [int(name.split("_")[-1]) for name in os.listdir(checkpoint_path) if name.startswith("version")]
        version = max(versions) if versions else None
        if version is None:
            raise ValueError("No versions found")
        path = f"{checkpoint_path}/version_{version}"
        for name, model in zip(("alpha", "rho", "volvol"), (self.z_alpha, self.z_rho, self.z_volvol)):
            model.load_checkpoint(f"{path}/{name}.pth")
        return self

    def save_checkpoint(self, checkpoint_path, version=None):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if version is None:
            versions = [int(name) for name in os.listdir(checkpoint_path) if name.isdigit()]
            version = max(versions) + 1 if versions else 0
        path = f"{checkpoint_path}/version_{version}"
        if not os.path.exists(path):
            os.makedirs(path)
        for name, model in zip(("alpha", "rho", "volvol"), (self.z_alpha, self.z_rho, self.z_volvol)):
            model.save_checkpoint(f"{path}/{name}.pth")

    def train(self, dataset: ScoresDataset, batch_size=1, num_workers=8, epochs=100):
        dataset_p = dataset.get_dataset("alpha")
        dataset_q = dataset.get_dataset("rho")
        dataset_r = dataset.get_dataset("volvol")

        dataloader_p = torch.utils.data.DataLoader(dataset_p, batch_size=1, num_workers=num_workers, shuffle=True)
        dataloader_q = torch.utils.data.DataLoader(dataset_q, batch_size=1, num_workers=num_workers, shuffle=True)
        dataloader_r = torch.utils.data.DataLoader(dataset_r, batch_size=1, num_workers=num_workers, shuffle=True)

        callback = EarlyStopping(
            monitor='train_loss',
            patience=3,
            verbose=True,
            mode='min',
            min_delta=0.01
        )

        print("Training alpha...")
        trainer = lightning.Trainer(max_epochs=epochs, callbacks=[callback])
        trainer.fit(self.z_alpha, dataloader_p)

        print("Training rho...")
        trainer = lightning.Trainer(max_epochs=epochs, callbacks=[callback])
        trainer.fit(self.z_rho, dataloader_q)

        print("Training volvol...")
        trainer = lightning.Trainer(max_epochs=epochs, callbacks=[callback])
        trainer.fit(self.z_volvol, dataloader_r)

        return self.z_alpha, self.z_rho, self.z_volvol

    def funcs(self):
        return {
            "alpha": self.z_alpha,
            "rho": self.z_rho,
            "volvol": self.z_volvol
        }

    def optim_candidates(self, rows, score_func, p=0.8):
        candidates = rows.iloc[:, 1:3].values

        inputs = torch.tensor(rows.iloc[:, 1:5].values[np.newaxis, :, :], dtype=torch.float32)
        scores = score_func(inputs)[0]  # get only first batch

        return SST.best_candidates(candidates, scores, p)

    def fit_params(self, inputs, p=0.8):
        score_funcs = self.funcs()

        return ParametricSABR.fit_params({
            key: self.optim_candidates(inputs[key], score_funcs[key], p) for key in ("alpha", "rho", "volvol")
        })
