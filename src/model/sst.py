import argparse
import os

import lightning
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from src.dataset import VolatilityDataset, Dataviewer
from src.sabr import ParametricSABR

from src.model.transformer import TransformerEncoder


class SST(nn.Module):
    def __init__(self,
                 n,
                 heads,
                 num_layers,
                 out_features=None):
        nn.Module.__init__(self)
        self.n = n
        self.transformer = TransformerEncoder(n, heads, num_layers, 4, 0.5, out_features)

    @staticmethod
    def get_combinations(point, candidates, k=1000, p=0.4):
        # param is alpha^{t} := (alpha, t)
        # candidates is a np.array of shape (n, 2), dim 0 = alpha, rho or volvol, dim 1 = tenor
        # generate k random combinations of size int(p * len(candidates)) that contain param[0]
        combination_size = int(p * len(candidates))

        # Generate k random combinations of size combination_size that contain point
        # Use numpy to get combinations without repetition in the same combination
        combinations = np.zeros((k, combination_size, 2))

        for i in range(k):
            indices = np.arange(len(candidates))
            combinations[i] = candidates[np.random.choice(indices, combination_size, replace=False)]
            combinations[i][np.random.randint(0, combination_size)] = point

        return combinations

    @classmethod
    def ss_value(cls, point, candidates, func, size, k=1000, p=0.4):
        sumation = 0

        combinations = cls.get_combinations(point, candidates, k, p)
        p_star = ParametricSABR.param_star(func, size, candidates)

        for combination in combinations:
            p_tilde = ParametricSABR.param_star(func, size, combination)

            sumation -= np.abs(
                func(point[1], p_tilde) - func(point[1], p_star)
            )

        return sumation / k

    @classmethod
    def z_values(cls, func, size, candidates, k=1000, p=0.4):

        ss_values = np.zeros(len(candidates))

        for idx, point in enumerate(candidates):
            ss_values[idx] = cls.ss_value(point, candidates, func, size, k, p)

        mean = np.mean(ss_values)
        std = np.std(ss_values)

        z_values = (ss_values - mean) / std

        return z_values

    @classmethod
    def z_alpha(cls, candidates, k=1000, p=0.4):
        return cls.z_values(ParametricSABR.alpha, 5, candidates, k, p)

    @classmethod
    def z_rho(cls, candidates, k=1000, p=0.4):
        return cls.z_values(ParametricSABR.rho, 4, candidates, k, p)

    @classmethod
    def z_volvol(cls, candidates, k=1000, p=0.4):
        return cls.z_values(ParametricSABR.volvol, 4, candidates, k, p)

    @classmethod
    def best_candidates(cls, inputs, values, p=0.4):
        # use values to return only the top p% of the inputs
        n = len(inputs)
        top = int((1 - p) * n)
        indices = torch.argsort(values, dim=0, descending=True)
        selected = indices[top:]
        return inputs[selected]

    def forward(self, x):
        return self.transformer(x)


class ParamDataset(Dataset):
    def __init__(self):
        self.dates = None
        self.data = None

    def load(self, params: str, path: str = "preprocessed/"):
        data = np.load(f"{path}{params}.npy")

        self.dates = np.unique(data[:, 0])
        self.data = data

    @staticmethod
    def get_values(func, size, raw_candidates, corrected_candidates, k, p):
        n = len(raw_candidates) + len(corrected_candidates)
        if len(corrected_candidates) == 0:
            candidates = np.array(raw_candidates)
        else:
            candidates = np.concatenate((raw_candidates, corrected_candidates), axis=0)
        inputs = np.zeros((n, 4))

        target = np.clip(SST.z_values(func, size, candidates, k, p), -1e9, 1e9)

        inputs[:, 0] = candidates[:, 1]
        inputs[:, 1] = candidates[:, 0]
        inputs[:, 2] = np.zeros(n)
        inputs[:, 2][:len(raw_candidates)] = 1

        prev_star = ParametricSABR.param_star(func, size, np.array(raw_candidates))

        for idx, point in enumerate(candidates):
            tenor, _ = point
            f = func(tenor, prev_star)
            if f is None:
                raise ValueError(f"Function {func.__name__} returned None for tenor {tenor}")
            else:
                inputs[idx, 3] = func(tenor, prev_star)

        # zero pad candidates to n
        inputs = np.pad(inputs, ((0, n - len(candidates)), (0, 0)))

        return inputs, target, candidates

    @classmethod
    def preprocess(cls, dataset: VolatilityDataset, path: str = "preprocessed/", k=1000, p=0.4):
        if not os.path.exists(path):
            os.makedirs(path)

        fixed_tenors = np.array([1 / 360, 7 / 360, 30 / 360, 60 / 360, 90 / 360, 180 / 360, 360 / 360])  # fixed = 7
        prev_day_params = {"p": None, "q": None, "r": None}

        alphas, rhos, volvols = [], [], []
        for idx, date in tqdm(enumerate(dataset.dates), total=len(dataset)):
            item = dataset[idx]
            tenors = np.unique(item[0][:, 2])

            raw_alphas, raw_rhos, raw_volvols = [], [], []
            corrected_alphas, corrected_rhos, corrected_volvols = [], [], []

            for tenor in tenors:
                x = item[0][item[0][:, 2] == tenor]
                y = item[1][item[0][:, 2] == tenor]
                alpha, beta, rho, volvol = SABRModel.fit(x, y)

                raw_alphas.append((alpha, tenor))
                raw_rhos.append((rho, tenor))
                raw_volvols.append((volvol, tenor))

            if prev_day_params["p"] is not None:
                for tenor in fixed_tenors:
                    corrected_alphas.append((ParametricSABR.alpha(tenor, prev_day_params["p"]), tenor))
                    corrected_rhos.append((ParametricSABR.rho(tenor, prev_day_params["q"]), tenor))
                    corrected_volvols.append((ParametricSABR.volvol(tenor, prev_day_params["r"]), tenor))

            inputs_alpha, target_alpha, candidates_alpha = cls.get_values(
                ParametricSABR.alpha, 5, raw_alphas, corrected_alphas, k, p)

            inputs_rho, target_rho, candidates_rho = cls.get_values(
                ParametricSABR.rho, 4, raw_rhos, corrected_rhos, k, p)

            inputs_volvol, target_volvol, candidates_volvol = cls.get_values(
                ParametricSABR.volvol, 4, raw_volvols, corrected_volvols, k, p)

            date = np.array([date] * len(inputs_alpha))
            tenors = np.concatenate((tenors, fixed_tenors)) if len(corrected_alphas) > 0 else tenors

            # if nan in alphas, rhos or volvols, skip
            if np.isnan(inputs_alpha).any() or np.isnan(inputs_rho).any() or np.isnan(inputs_volvol).any():
                continue

            alphas.append(
                np.concatenate((date[:, np.newaxis], tenors[:, np.newaxis], inputs_alpha, target_alpha[:, np.newaxis]),
                               axis=1))
            rhos.append(
                np.concatenate((date[:, np.newaxis], tenors[:, np.newaxis], inputs_rho, target_rho[:, np.newaxis]),
                               axis=1))
            volvols.append(np.concatenate(
                (date[:, np.newaxis], tenors[:, np.newaxis], inputs_volvol, target_volvol[:, np.newaxis]), axis=1))

            prev_day_params["p"] = ParametricSABR.p_star(candidates_alpha)
            prev_day_params["q"] = ParametricSABR.q_star(candidates_rho)
            prev_day_params["r"] = ParametricSABR.r_star(candidates_volvol)

        def save(candidates, name):
            candidates = np.vstack(candidates)
            numerical_candidates = candidates[:, 1:].astype(np.float32)
            candidates = candidates[~np.isnan(numerical_candidates).any(axis=1)]
            numerical_candidates = candidates[:, 1:].astype(np.float32)
            candidates = candidates[~np.isinf(numerical_candidates).any(axis=1)]

            np.save(f"{path}{name}.npy", candidates)

        save(alphas, "alphas")
        save(rhos, "rhos")
        save(volvols, "volvols")

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]

        points = self.data[self.data[:, 0] == date][:, 1:].astype(np.float32)

        return (torch.tensor(points[:, :-1], dtype=torch.float32),
                torch.tensor(points[:, -1], dtype=torch.float32))


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
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = torch.clamp(torch.tensor(batch[0][:, :, 1:], dtype=torch.float32), -1e9, 1e9)
        y = batch[1]
        output = self.model(x)

        loss = self.criterion(output[:, :, 0], y)

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


def main(preprocess=False, checkpoint=None):
    if checkpoint is not None:
        checkpoint_paths = {i: f"{checkpoint}_{i}.pth" for i in ("alpha", "rho", "volvol")}
    else:
        checkpoint_paths = {"alpha": None, "rho": None, "volvol": None}
    sst_p = LitSST(4, 2, 4, 1, checkpoint_paths["alpha"], lr=1e-5)
    sst_q = LitSST(4, 2, 4, 1, checkpoint_paths["rho"], lr=1e-6)
    sst_r = LitSST(4, 2, 4, 1, checkpoint_paths["volvol"], lr=1e-3)

    dataset = VolatilityDataset()
    dataset.load("option_SPY_dataset_combined.csv")
    data_p, data_q, data_r = ParamDataset(), ParamDataset(), ParamDataset()

    if preprocess:
        data_p.preprocess(dataset, "preprocessed/", k=2)

    data_p.load("alphas")
    data_q.load("rhos")
    data_r.load("volvols")

    def train(lit_sst, data, epochs=100):
        trainer = lightning.Trainer(max_epochs=epochs)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
        trainer.fit(lit_sst, dataloader)

    def get_star(lit_sst, data, func, size):
        scores = lit_sst.model(data[0][0][np.newaxis, :, :-1])
        best_candidates = SST.best_candidates(data[0][0][:, :-1], scores[0][:, 0], p=0.4).detach().numpy()
        return ParametricSABR.param_star(func, size, best_candidates)

    if checkpoint is None or not all([os.path.exists(checkpoint_paths[i]) for i in ("alpha", "rho", "volvol")]):
        train(sst_p, data_p)
        train(sst_q, data_q)
        train(sst_r, data_r)

    p_star = get_star(sst_p, data_p, ParametricSABR.alpha, 5)
    q_star = get_star(sst_q, data_q, ParametricSABR.rho, 4)
    r_star = get_star(sst_r, data_r, ParametricSABR.volvol, 4)

    if checkpoint is not None and not all([os.path.exists(checkpoint_paths[i]) for i in ("alpha", "rho", "volvol")]):
        sst_p.save_checkpoint(f"{checkpoint}_alpha.pth")
        sst_q.save_checkpoint(f"{checkpoint}_rho.pth")
        sst_r.save_checkpoint(f"{checkpoint}_volvol.pth")

    if sst_p.losses:
        losses = [np.array(loss) for loss in [sst_p.losses, sst_q.losses, sst_r.losses]]
        # apply moving window mean
        window = 10
        losses = [np.convolve(loss, np.ones(window) / window, mode='valid') for loss in losses]

        # plot losses
        plt.plot(losses[0], label="Alpha")
        plt.plot(losses[1], label="Rho")
        plt.plot(losses[2], label="Volvol")
        plt.legend()
        plt.savefig("losses.png")

        plt.show()

    # get 20 spaced points from K and T
    rf = dataset.data["r"].values[0]
    div = dataset.data["d"].values[0]

    sabr = ParametricSABR(rf, div)

    # plot 3d surface scatter
    viewer = Dataviewer()
    last_date = dataset.dates[-1]
    real_surface = dataset.get((-np.inf, np.inf), (-np.inf, np.inf), last_date)
    maturities = np.unique(real_surface["maturity"].values)
    maturities = np.linspace(maturities[0], maturities[-1], 20)
    strikes = np.unique(real_surface["strike"].values)
    strikes = np.linspace(strikes[0], strikes[-1], 20)
    spot = dataset.data["underlying"].values[-1]

    real_surface = real_surface[['strike', 'maturity', 'iv']]
    viewer.plot(real_surface)
    plt.savefig("real_surface.png")

    beta = 0.1
    pred_surface = sabr.smooth_surface(spot, maturities, strikes, {"p": p_star, "q": q_star, "r": r_star}, beta)

    # surface is a matrix, transform to df.columns = ["strike", "maturity", "iv"]
    strikes_grid, maturities_grid = np.meshgrid(strikes, maturities, indexing='ij')
    pred_surface = pd.DataFrame({
        "strike": strikes_grid.ravel(),
        "maturity": maturities_grid.ravel(),
        "iv": pred_surface.ravel()
    })
    pred_surface.columns = ["strike", "maturity", "iv"]

    # plot scatter
    viewer.plot(pred_surface)
    plt.savefig("pred_surface.png")
    plt.show()


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument('--checkpoint', type=str, help='Path to save/load model checkpoint')

    args = parser.parse_args()

    main(args.preprocess, args.checkpoint)
