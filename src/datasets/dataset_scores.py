import os
import warnings
from time import time

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .dataset_vol import VolatilityDataset
from ..sabr import SABR, ParametricSABR


class SSV:
    @staticmethod
    def mc_combinations(point, candidate_group, k=50, p=0.8):
        """
        Monte Carlo combinations of candidate groups, that contain given point

        Args:
            point: point to be present in the combinations
            candidate_group: group of candidates out of which the combinations are drawn
            k: number of combinations for estimation, the larger the better but slower
            p: subset size, the smaller the subset, the larger the summarization, but higher variance
        """
        # subset size, to simulate summarization
        # a smaller subset size increases the variance of the error, but reduces the bias
        # this is done to implicitly remove outliers
        n = int(p * len(candidate_group))

        indices = np.arange(len(candidate_group))
        combinations = np.array([candidate_group[np.random.choice(indices, n, replace=False)] for _ in range(k)])

        combinations[:, np.random.randint(0, n)] = point  # For each combination, insert the point at random index

        return combinations

    @classmethod
    def summary_score_value(cls, candidates, optim_func, func, k=20, p=0.8, sigma=1e-1):
        """
        Summary Score Value (SSV) for a given candidate group

        Args:
            candidates: group of candidates
            optim_func: function to optimize for the best parameter
            func: function to calculate the resulting value from the parameter
            k: number of combinations for estimation, the larger the better but slower
            p: subset size, the smaller the subset, the larger the summarization, but higher variance
            sigma: scaling factor for the error
        """
        # Best possible approximation to parameter (P, Q or R) maintaining all candidates
        param_star = optim_func(candidates)

        ss_values = np.zeros(len(candidates))

        for idx, candidate in enumerate(candidates):
            combinations = cls.mc_combinations(candidate, candidates, k, p)

            # Parameter (P, Q or R) summarized from smaller combinations
            param_tilde = np.array([optim_func(combinations[i]) for i in range(k)])

            # MAE
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                error = np.mean(np.abs([func(candidate[0], param) - candidate[1] for param in param_tilde])) / k

            # The closer the error to 0, the better the approximation, a perfect score is 1
            ss_values[idx] = np.sqrt(np.exp(-error / sigma))

            if np.isnan(ss_values[idx]):
                ss_values[idx] = 0

        # Z-normalization
        ss_values = (ss_values - ss_values.mean()) / ss_values.std()

        return ss_values, param_star

    @classmethod
    def get_funcs(cls, param, *args, **kwargs):
        funcs = {
            "alpha": lambda candidates: cls.summary_score_value(
                candidates, ParametricSABR.fit_p, ParametricSABR.alpha, *args, **kwargs),

            "rho": lambda candidates: cls.summary_score_value(
                candidates, ParametricSABR.fit_q, ParametricSABR.rho, *args, **kwargs),

            "volvol": lambda candidates: cls.summary_score_value(
                candidates, ParametricSABR.fit_r, ParametricSABR.volvol, *args, **kwargs)
        }
        return funcs[param]


class ScoresDataset:
    def __init__(self, volatility_dataset: VolatilityDataset = None):
        self.volatility_dataset = volatility_dataset if volatility_dataset is not None else VolatilityDataset()
        self.dataset = {}
        self._load()

    def _load(self):
        # test if alpha.csv, rho.csv and volvol.csv exist
        if not os.path.exists(f"{self.volatility_dataset.path}"):
            os.makedirs(f"{self.volatility_dataset.path}")

        if not os.path.exists(f"{self.volatility_dataset.path}/alpha.csv"):
            print("Preprocessing dataset...")
            self.dataset = self.preprocess()
            for key in ["alpha", "rho", "volvol"]:
                self.dataset[key].to_csv(f"{self.volatility_dataset.path}/{key}.csv", index=False)
            return self.dataset
        else:
            for key in ["alpha", "rho", "volvol"]:
                self.dataset[key] = pd.read_csv(f"{self.volatility_dataset.path}/{key}.csv")
            return self.dataset

    def preprocess(self):
        # get top 7 most common maturities
        fixed_maturities = self.volatility_dataset.data["maturity"].value_counts().index[:7]
        columns = ["day", "maturity", "value", "is_raw", "prev_value", "target"]

        dataset = {
            "alpha": pd.DataFrame(columns=columns),
            "rho": pd.DataFrame(columns=columns),
            "volvol": pd.DataFrame(columns=columns)
        }

        p, q, r = None, None, None

        def get_prev_value(t):
            values = {"alpha": None, "rho": None, "volvol": None}

            if p is not None:
                params = {
                    "alpha": p,
                    "rho": q,
                    "volvol": r
                }

                for key in ["alpha", "rho", "volvol"]:
                    values[key] = ParametricSABR.funcs()[key](t, params[key])

            return values

        def fit_maturity(ivol, S, K, t, rf, div, is_raw=1):
            alpha, beta, rho, volvol = SABR.fit_sabr(ivol, S, K, t, rf, div)
            values = {
                "alpha": alpha,
                "rho": rho,
                "volvol": volvol
            }

            if alpha is None or rho is None or volvol is None:
                return values, None

            prev_values = get_prev_value(t)

            row = {key: [day, t, values[key], is_raw, prev_values[key], None] for key in ["alpha", "rho", "volvol"]}

            return values, row

        for day in range(len(self.volatility_dataset)):
            start_time = time()
            S, K, T, rf, div, ivol = self.volatility_dataset.get(day)
            maturities = np.unique(np.concatenate((T, fixed_maturities)))

            candidates = {
                "alpha": [],
                "rho": [],
                "volvol": []
            }

            for t in np.unique(T):
                values, row = fit_maturity(ivol[T == t], S, K[T == t], t, rf, div, is_raw=1)

                if row is None:
                    continue

                for key in ["alpha", "rho", "volvol"]:
                    candidates[key].append((t, values[key]))
                    dataset[key].loc[len(dataset[key])] = row[key]

            for t in maturities:
                if t in np.unique(T):
                    continue

                values = get_prev_value(t)
                row = {key: [day, t, values[key], 0, values[key], None] for key in ["alpha", "rho", "volvol"]}

                if None in values.values():
                    continue

                for key in ["alpha", "rho", "volvol"]:
                    candidates[key].append((t, values[key]))
                    dataset[key].loc[len(dataset[key])] = row[key]

            for key in ["alpha", "rho", "volvol"]:
                func = SSV.get_funcs(key, k=10, p=0.8, sigma=1)
                candidates[key] = np.array(candidates[key])

                scores, param_star = func(candidates[key])

                # print error if any score is None
                if None in scores:
                    print(f"Error in day {day} for {key}")
                dataset[key].loc[dataset[key]["day"] == day, "target"] = scores

                if key == "alpha":
                    p = param_star
                elif key == "rho":
                    q = param_star
                elif key == "volvol":
                    r = param_star

            expected_remaining_time = int((time() - start_time) * (len(self.volatility_dataset) - day) / 60)
            print(f"Day {day}/{len(self.volatility_dataset)} - "
                  f"Expected Remaining Time: {expected_remaining_time} minutes")

        # Drop first day rows
        for key in ["alpha", "rho", "volvol"]:
            dataset[key] = dataset[key].iloc[1:]

        return dataset

    def __len__(self):
        return len(self.volatility_dataset)

    def get_dataset(self, key):
        dataset = self.dataset[key]

        class ValuesDataset(Dataset):
            def __init__(self):
                self.data = dataset
                self.dates = self.data["day"].unique()

            def __len__(self):
                return len(self.dates)

            def __getitem__(self, idx):
                date = self.dates[idx]
                rows = self.data[self.data["day"] == date].values.astype(np.float32)
                x = rows[:, 1:5]
                y = rows[:, 5]
                return x, y

        return ValuesDataset()
