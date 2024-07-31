import argparse

import numpy as np

from src.datasets.dataset_scores import SSVDataset
from src.datasets.dataset_vol import VolatilityDataset, Dataviewer
from src.models.sst import MultiSST
from src.sabr import ParametricSABR


def preprocess():
    volatility_dataset = VolatilityDataset("dataset").load("option_SPY_dataset_combined.csv")
    return SSVDataset("dataset").load(volatility_dataset), volatility_dataset


def test(checkpoint, pos):
    viewer = Dataviewer()
    ssv_dataset, volatility_dataset = preprocess()
    model = MultiSST(4, 4, 32, 32, forward_expansion=256)

    n = 1
    fig, axs = viewer.create_grid(2, n)

    def sample_test(idx):
        sample = ssv_dataset.sample(pos + idx)

        p, q, r = model.fit_params(sample, p=0.7)
        # standardize r
        p = (p - p.mean()) / p.std()
        print(f"{pos+idx} - p: {p}, q: {q}, r: {r}")

        sabr = ParametricSABR(p, q, r)

        _, S, K, T, rf, div = volatility_dataset.sample(pos + idx)
        true_surface = volatility_dataset.get((-np.inf, np.inf), (-np.inf, np.inf), volatility_dataset.dates[pos + idx])
        viewer.plot(true_surface[['strike', 'maturity', 'iv']], axs[0, idx] if n > 1 else axs[0])

        K = np.linspace(K.min(), K.max(), 50)
        T = np.linspace(T.min(), T.max(), 50)
        pred_ivol = sabr.smooth_surface(S, K, T, rf, div, beta=0.5)

        viewer.plot_ravel(K, T, pred_ivol, axs[1, idx] if n > 1 else axs[1])

    for i in range(n):
        sample_test(i)

    viewer.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to save/load models checkpoint', default="weights")

    args = parser.parse_args()
    test(args.checkpoint, np.random.randint(0, 68 - 5))
