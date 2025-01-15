import os

import lightning
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn, autograd
from torch.utils.data import Dataset

from .transformer import TransformerEncoder
from ..datasets import ScoresDataset
from ..sabr import ParametricSABR

import torch


def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, seq_len, num_features = x0.shape
    X = torch.zeros(bsz, m, seq_len * num_features, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, seq_len * num_features, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view(bsz, seq_len, num_features)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    k = 1
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].view(bsz, seq_len, num_features)).view(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))
        if res[-1] < tol:
            break

    return X[:, k % m].view(bsz, seq_len, num_features), res


class DEQFixedPoint(nn.Module):
    def __init__(self, in_features, solver=anderson, **kwargs):
        super().__init__()
        self.solver = solver
        self.kwargs = kwargs
        self.layer = nn.Linear(in_features, in_features)
        self.norm = nn.LayerNorm(in_features)
        self.relu = nn.ReLU()
        self.x = None

    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            y = torch.zeros_like(x)
            z, self.x = self.solver(lambda z: self.f(z, x), y, **self.kwargs)
        z = self.f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g

        z.register_hook(backward_hook)
        return z

    def f(self, x, y):
        return self.norm(self.relu(self.layer(x) + y))


class SST(nn.Module):
    def __init__(self,
                 in_features,
                 heads,
                 num_blocks,
                 num_layers,
                 forward_expansion=4,
                 fc_size=256,
                 out_features=1,
                 dropout=0.2):
        nn.Module.__init__(self)
        self.transformer = TransformerEncoder(in_features, heads, num_blocks, num_layers, forward_expansion,
                                              out_features, dropout)
        self.fc_size = fc_size
        self.feed_forward = nn.Sequential(
            nn.Linear(forward_expansion, self.fc_size),
            nn.ReLU(),
            nn.Linear(self.fc_size, self.fc_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(self.fc_size // 2),
            nn.Linear(self.fc_size // 2, out_features),
        )

        self.deq_forward = nn.Sequential(
            DEQFixedPoint(self.fc_size // 2, tol=1e-2, max_iter=50, lam=1e-4, beta=1.0),
            nn.Dropout(0.2),
            nn.LayerNorm(self.fc_size // 2),
            nn.Linear(self.fc_size // 2, self.fc_size // 4),
            DEQFixedPoint(self.fc_size // 4, tol=1e-2, max_iter=50, lam=1e-4, beta=1.0),
            nn.Dropout(0.2),
            nn.LayerNorm(self.fc_size // 4),
            nn.Linear(self.fc_size // 4, out_features),
        )

    @classmethod
    def best_candidates(cls, values, scores, p=0.8):
        # use values to return only the top p% of the inputs
        n = len(values)
        top = int((1 - p) * n)
        indices = torch.argsort(scores, dim=0, descending=False)
        selected = indices[top:]
        return values[selected]

    def forward(self, x):
        x = self.transformer(x)
        x = self.feed_forward(x)
        return x


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

        if loss is None or np.isnan(loss.item()) or np.isinf(loss.item()):
            raise ValueError(f"Loss is None for batch {batch_idx}")

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.losses.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)

        loss = self.criterion(output, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

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
                 fc_size=256,
                 out_features=1,
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
            LitSST(checkpoint_path, lr, in_features, heads, num_blocks, num_layers, forward_expansion, fc_size,
                   out_features,
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

    def train(self, dataset: ScoresDataset, num_workers=8, epochs=100):
        dataset_p = dataset.get_dataset("alpha")
        dataset_q = dataset.get_dataset("rho")
        dataset_r = dataset.get_dataset("volvol")

        dataloader_p = torch.utils.data.DataLoader(dataset_p, batch_size=1, num_workers=num_workers, shuffle=True)
        dataloader_q = torch.utils.data.DataLoader(dataset_q, batch_size=1, num_workers=num_workers, shuffle=True)
        dataloader_r = torch.utils.data.DataLoader(dataset_r, batch_size=1, num_workers=num_workers, shuffle=True)

        for sst, dataloader in zip((self.z_alpha, self.z_rho, self.z_volvol),
                                   (dataloader_p, dataloader_q, dataloader_r)):
            callback = EarlyStopping(
                monitor='train_loss',
                patience=10,
                verbose=True,
                mode='min',
                min_delta=0.5e-2
            )

            trainer = lightning.Trainer(max_epochs=epochs, callbacks=[callback], accelerator="cpu")
            trainer.fit(sst, dataloader)

        return self.z_alpha, self.z_rho, self.z_volvol

    @property
    def funcs(self):
        return {
            "alpha":  self.z_alpha,
            "rho":    self.z_rho,
            "volvol": self.z_volvol
        }

    def get_optimal_parameters(self, rows: dict, p=0.8):
        optimal = {}

        for key in ["alpha", "rho", "volvol"]:
            candidates = rows[key][0][:, 0:2]
            scores = self.funcs[key](torch.tensor(rows[key][0][np.newaxis, ...], dtype=torch.float32))
            optimal[key] = SST.best_candidates(candidates, scores[0], p)

        p, q, r = ParametricSABR.fit_params(optimal)

        return ParametricSABR(p, q, r), optimal
