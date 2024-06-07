import logging
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
from tqdm import tqdm

from src.data import Dataviewer, VolatilityDataset
from src.sabr import SABR, ParametricSABR


class MyTestCase(unittest.TestCase):
    def sample_surface(self):
        dataset = VolatilityDataset("../dataset")
        dataset.load("../dataset/option_SPY_dataset_combined.csv")

        return dataset[0]

    def test_parametric(self):
        values, target = self.sample_surface()
        S = values[0, 0]
        K = values[:, 1]
        T = values[:, 2]
        rf = values[0, 3]
        div = values[0, 4]

        real_iv = target

        Dataviewer.plot(pd.DataFrame({
            "strike": K,
            "maturity": T,
            "iv": real_iv
        }))

        # (5,)
        p = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        q = np.array([0.1, 0.1, 0.1, 0.1])
        r = np.array([0.1, 0.1, 0.1, 0.1])

        sabr = ParametricSABR(p, q, r)
        beta = 0.5

        K = np.linspace(K.min(), K.max(), 20)
        T = np.linspace(T.min(), T.max(), 20)

        iv = sabr.smooth_surface(S, K, T, rf=rf, div=div, beta=beta)
        Dataviewer.plot_ravel(K, T, iv)

    def test_pstar(self):
        values, target = self.sample_surface()
        S = values[0, 0]
        K = values[:, 1]
        T = values[:, 2]
        rf = values[0, 3]
        div = values[0, 4]

        alpha_candidates, rho_candidates, volvol_candidates = [], [], []

        logger = logging.getLogger(__name__)

        for idx, t in tqdm(enumerate(T), total=len(T)):
            mask = values[:, 2] == t
            K = values[mask, 1]
            ivol = target[mask]

            try:
                alpha, beta, rho, volvol = SABR.fit_sabr(ivol, S, K, t, rf, div)
            except OptimizeWarning:
                logger.warning(f"Optimization failed at {t}")
                continue

            alpha_candidates.append(alpha)
            rho_candidates.append(rho)
            volvol_candidates.append(volvol)

        p, q, r = ParametricSABR.fit_params({
            "alpha": alpha_candidates,
            "rho": rho_candidates,
            "volvol": volvol_candidates
        })

        sabr = ParametricSABR(p, q, r)
        beta = 0.5

        K = np.linspace(K.min(), K.max(), 20)
        T = np.linspace(T.min(), T.max(), 20)

        iv = sabr.smooth_surface(S, K, T, rf=rf, div=div, beta=beta)
        Dataviewer.plot_ravel(K, T, iv)


if __name__ == '__main__':
    unittest.main()
