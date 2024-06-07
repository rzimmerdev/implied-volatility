import logging
import unittest

import numpy as np
import pandas as pd

from src.dataset import Dataviewer, VolatilityDataset
from src.sabr import ParametricSABR


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

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        ivol = target
        beta = ParametricSABR.optim_beta(ivol, S, K, T, rf, div, None, logger)
        print(f"Best beta: {beta}")

        candidates = ParametricSABR.fit_candidates(ivol, S, K, T, rf, div, beta, logger)
        p, q, r = ParametricSABR.fit_params(candidates)
        sabr = ParametricSABR(p, q, r)
        sabr.save(beta)

        Dataviewer.plot(pd.DataFrame({
            "strike": K,
            "maturity": T,
            "iv": ivol
        }))

        K = np.linspace(K.min(), K.max(), 20)
        T = np.linspace(T.min(), T.max(), 20)
        pred_ivol = sabr.smooth_surface(S, K, T, rf=rf, div=div, beta=beta)

        Dataviewer.plot_ravel(K, T, pred_ivol)


if __name__ == '__main__':
    unittest.main()
