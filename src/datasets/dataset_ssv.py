import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from .dataset_vol import VolatilityDataset
from ..models import SSV
from ..sabr import ParametricSABR


class SSVDataset:
    def __init__(self, path="dataset"):
        super().__init__()
        self.path = path

        self._vol_dataset: VolatilityDataset | None = None
        self.data = None

    def sample(self, idx=0):
        return {
            "alpha": self.data["alpha"][self.data["alpha"]["date"] == idx],
            "rho": self.data["rho"][self.data["rho"]["date"] == idx],
            "volvol": self.data["volvol"][self.data["volvol"]["date"] == idx]
        }

    def _load(self):
        # read alpha, rho, volvol from csv files
        self.data = {
            "alpha": pd.read_csv(f"{self.path}/dataset_alpha.csv"),
            "rho": pd.read_csv(f"{self.path}/dataset_rho.csv"),
            "volvol": pd.read_csv(f"{self.path}/dataset_volvol.csv")
        }

    def load(self, file="archive", volatility_dataset=None):
        """
        Loads dataset/dataset_param.csv,
        if any doesn't exist, run preprocessing on volatility_dataset
        """
        if not os.path.exists(f"{self.path}/dataset_alpha.csv") or \
                not os.path.exists(f"{self.path}/dataset_rho.csv") or \
                not os.path.exists(f"{self.path}/dataset_volvol.csv"):

            if volatility_dataset is not None:
                self._vol_dataset = volatility_dataset
            else:
                self._vol_dataset = VolatilityDataset(self.path, file).load("option_SPY_dataset_combined.csv")
            self._preprocess()
        else:
            self._load()

        return self

    def _preprocess_day(self, idx, beta=0.4):
        ivol, S, K, T, rf, div = self._vol_dataset.sample(idx)

        candidates = ParametricSABR.candidates(ivol, S, K, T, rf, div, beta)

        z_alpha, p = SSV.z_alpha(candidates["alpha"], k=10, p=0.7)
        z_rho, q = SSV.z_rho(candidates["rho"], k=10, p=0.7)
        z_volvol, r = SSV.z_volvol(candidates["volvol"], k=10, p=0.7)

        return candidates, {"alpha": z_alpha, "rho": z_rho, "volvol": z_volvol}, (p, q, r)

    def _sample_params(self) -> dict:
        idx = 0
        ivol, S, K, T, rf, div = self._vol_dataset.sample(idx)
        p, q, r, _ = ParametricSABR.optim(ivol, S, K, T, rf, div)
        return {
            "alpha": p,
            "rho": q,
            "volvol": r
        }

    @classmethod
    def get_inputs(cls, idx, z_func, func, candidates, fixed_maturities, prev_param):
        i = 0
        for maturity in fixed_maturities:
            if maturity not in candidates[:, 0]:
                candidates = np.vstack(
                    [candidates, (maturity, func(maturity, prev_param))])
                i += 1

        rows = np.zeros((len(candidates), 6))

        rows[:, 0] = idx  # date
        rows[:, 1] = candidates[:, 0]  # maturity
        rows[:, 2] = candidates[:, 1]  # candidate value
        rows[:, 3] = np.zeros(len(candidates))
        rows[i:, 3] = 1
        rows[:, 4] = func(rows[:, 1], prev_param)
        rows[:, 5] = z_func(candidates)[0]

        return rows

    def _preprocess(self):
        input_columns = ["date", "maturity", "value", "is_prev", "prev_day_value"]
        fixed_maturities = self._vol_dataset.data["maturity"].value_counts().head(7).index.values
        output_columns = ["z_score"]

        prev_params = self._sample_params()

        self.data = {
            "alpha": pd.DataFrame(columns=input_columns + output_columns),
            "rho": pd.DataFrame(columns=input_columns + output_columns),
            "volvol": pd.DataFrame(columns=input_columns + output_columns)
        }

        funcs = ParametricSABR.funcs()
        z_funcs = SSV.z_funcs()

        for idx in tqdm(range(len(self._vol_dataset))):
            with np.errstate(all='ignore'):
                candidates, z_scores, current_params = self._preprocess_day(idx)

                for key in self.data.keys():
                    rows = self.get_inputs(idx, z_funcs[key], funcs[key], candidates[key], fixed_maturities, prev_params[key])
                    self.data[key] = pd.concat([self.data[key], pd.DataFrame(rows, columns=input_columns + output_columns)])

        # save to csv: alpha.csv, rho.csv, volvol.csv
        for key in self.data.keys():
            self.data[key].to_csv(f"{self.path}dataset_{key}.csv", index=False)
