import os

import lightning
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.data import VolatilityDataset
from src.sabr.sabr import SABRModel

from src.model.transformer import TransformerEncoder
from src.sabr.ssv import SSV


class SST(SSV, nn.Module):
    def __init__(self,
                 n,
                 heads,
                 num_layers,
                 out_features=None,
                 rf=None,
                 dv=None):
        SSV.__init__(self, rf, dv)
        nn.Module.__init__(self)
        self.n = n
        self.transformer = TransformerEncoder(n, heads, num_layers, 4, 0.5, out_features)

    def get_values(self, func, size, raw_candidates, corrected_candidates, k, p):
        n = len(raw_candidates) + len(corrected_candidates)
        if len(corrected_candidates) == 0:
            candidates = np.array(raw_candidates)
        else:
            candidates = np.concatenate((raw_candidates, corrected_candidates), axis=0)
        inputs = np.zeros((n, 4))

        param_star = self.sabr.param_star(func, size, candidates)

        target = np.array(
            [func(tenor, param_star) for tenor in candidates[:, 1]]
        )

        z_values = self.z_values(func, size, candidates, k, p)

        inputs[:, 0] = z_values
        inputs[:, 1] = candidates[:, 0]
        inputs[:, 2] = np.zeros(n)
        inputs[:, 2][:len(raw_candidates)] = 1

        prev_star = self.sabr.param_star(func, size, np.array(raw_candidates))

        for idx, point in enumerate(candidates):
            tenor, _ = point
            inputs[idx, 3] = func(tenor, prev_star)

        # zero pad candidates to n
        inputs = np.pad(inputs, ((0, n - len(candidates)), (0, 0)))

        return inputs, target, candidates

    def best_candidates(self, inputs, values, p=0.4):
        # use values to return only the top p% of the inputs
        n = len(inputs)
        top = int((1 - p) * n)
        indices = torch.argsort(values, descending=True)
        selected = indices[top:]
        return inputs[selected]

    def forward(self, x):
        return self.transformer(x)

    def param_star(self, func, size, candidates):
        return self.sabr.param_star(func, size, candidates)


class ParamDataset(Dataset):
    def __init__(self, sst=None):
        self.dates = None
        self.data = None
        self.sst = None

    def load(self, params: str = "alphas", path: str = "preprocessed/"):
        data = np.load(f"{path}{params}.npy")

        self.dates = np.unique(data[:, 0])
        self.data = data

    def preprocess(self, sst, dataset: VolatilityDataset, path: str = "preprocessed/", k=1000, p=0.4):
        if not os.path.exists(path):
            os.makedirs(path)

        sabr = sst.sabr
        fixed_tenors = np.array([1 / 360, 7 / 360, 30 / 360, 60 / 360, 90 / 360, 180 / 360, 360 / 360])  # fixed = 7
        prev_day_params = {"p": None, "q": None, "r": None}

        alphas, rhos, volvols = [], [], []
        for idx, date in enumerate(dataset.dates[:3]):
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
                    corrected_alphas.append((sabr.alpha(tenor, prev_day_params["p"]), tenor))
                    corrected_rhos.append((sabr.rho(tenor, prev_day_params["q"]), tenor))
                    corrected_volvols.append((sabr.volvol(tenor, prev_day_params["r"]), tenor))

            inputs_alpha, target_alpha, candidates_alpha = sst.get_values(sabr.alpha, 5, raw_alphas, corrected_alphas, k, p)
            inputs_rho, target_rho, candidates_rho = sst.get_values(sabr.rho, 4, raw_rhos, corrected_rhos, k, p)
            inputs_volvol, target_volvol, candidates_volvol = sst.get_values(sabr.volvol, 4, raw_volvols, corrected_volvols, k, p)

            date = np.array([date] * len(inputs_alpha))
            tenors = np.concatenate((tenors, fixed_tenors)) if len(corrected_alphas) > 0 else tenors
            alphas.append(np.concatenate((date[:, np.newaxis], tenors[:, np.newaxis], inputs_alpha, target_alpha[:, np.newaxis]), axis=1))
            rhos.append(np.concatenate((date[:, np.newaxis], tenors[:, np.newaxis], inputs_rho, target_rho[:, np.newaxis]), axis=1))
            volvols.append(np.concatenate((date[:, np.newaxis], tenors[:, np.newaxis], inputs_volvol, target_volvol[:, np.newaxis]), axis=1))

            prev_day_params["p"] = sabr.p_star(candidates_alpha)
            prev_day_params["q"] = sabr.q_star(candidates_rho)
            prev_day_params["r"] = sabr.r_star(candidates_volvol)

        alphas = np.vstack(alphas)
        rhos = np.vstack(rhos)
        volvols = np.vstack(volvols)

        np.save(f"{path}alphas.npy", alphas)
        np.save(f"{path}rhos.npy", rhos)
        np.save(f"{path}volvols.npy", volvols)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]

        candidates = self.data[self.data[:, 0] == date][:, 1:].astype(np.float32)

        return (torch.tensor(candidates[:, :-1], dtype=torch.float32),
                torch.tensor(candidates[:, -1], dtype=torch.float32))


class LitSST(lightning.LightningModule):
    def __init__(self, in_features, heads, num_layers, out_features=None, rf=None, dv=None):
        super(LitSST, self).__init__()
        self.model = SST(in_features, heads, num_layers, out_features, rf, dv)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = torch.tensor(batch[0][:, :, 1:], dtype=torch.float32)
        y = torch.tensor(batch[1], dtype=torch.float32)
        # pad with zeroes to in_features
        # x = torch.cat((x, torch.zeros((x.shape[0], self.model.n - x.shape[1], x.shape[2]), device=x.device)), dim=1)
        # y = torch.cat((y, torch.zeros((y.shape[0], self.model.n - y.shape[1]), device=y.device)), dim=1)
        output = self.model(x)
        loss = self.criterion(output, y)
        return loss

    def configure_optimizers(self):
        return self.optimizer


def main():
    lit_sst = LitSST(4, 2, 4, 1, 0.05, 0.02)

    dataset = VolatilityDataset()
    dataset.load("option_SPY_dataset_combined.csv")

    param_dataset = ParamDataset(sst=lit_sst.model)
    # param_dataset.preprocess(lit_sst.model, dataset, k=2)
    param_dataset.load()

    trainer = lightning.Trainer(max_epochs=10)
    dataloader = torch.utils.data.DataLoader(param_dataset, batch_size=1, shuffle=True)
    trainer.fit(lit_sst, dataloader)
    model = lit_sst.model
    best_candidates = model.best_candidates(param_dataset[0][0][:, 0:2], param_dataset[0][1])
    p_star = model.param_star(model.sabr.alpha, 5, best_candidates.numpy())


if __name__ == "__main__":
    main()
