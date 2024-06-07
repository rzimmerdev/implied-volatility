import unittest

import numpy as np
import matplotlib.pyplot as plt

from src.dataset import Dataviewer, VolatilityDataset
from src.sabr import SABR


class MyTestCase(unittest.TestCase):
    def test_ivol(self):
        S = 100
        K = np.linspace(80, 120, 20)
        T = np.linspace(0.1, 1, 20)

        alpha = 0.1
        beta = 0.5
        rho = 0.1
        volvol = 0.1

        r = 0.01
        d = 0.0

        iv = np.zeros((len(T), len(K)))
        for idx, t in enumerate(T):
            iv[idx] = SABR.ivol(alpha, beta, rho, volvol, S, K, t, r, d)

        Dataviewer.plot_ravel(K, T, iv)

    def sample_surface(self):
        dataset = VolatilityDataset("../dataset")
        dataset.load("../dataset/option_SPY_dataset_combined.csv")

        return dataset[0]

    def test_fit_sabr(self):
        values, target = self.sample_surface()

        rf = values[0, 3]
        div = values[0, 4]

        n = len(values)
        S = values[n // 2, 0]

        maturity = values[:, 2]
        t = maturity[0]

        mask = values[:, 2] == t
        K = values[mask, 1]
        ivol = target[mask]

        # add legend
        plt.plot(K, ivol, 'x', label='target')

        beta = None
        alpha, beta, rho, volvol = SABR.fit_sabr(ivol, S, K, t, rf, div, beta)
        print(alpha, beta, rho, volvol)

        ivol = SABR.ivol(alpha, beta, rho, volvol, S, K, t, rf, div)

        plt.plot(K, ivol, 'x', label='fitted')
        plt.legend()
        plt.show()

        err = np.sum((np.subtract(ivol, target[mask])) ** 2)
        print(err)


if __name__ == '__main__':
    unittest.main()
