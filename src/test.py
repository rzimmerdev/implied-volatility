import argparse

import numpy as np

from src.datasets.dataset_ssv import SSVDataset
from src.datasets.dataset_vol import VolatilityDataset, Dataviewer
from src.models.sst import MultiSST
from src.sabr import ParametricSABR


def preprocess():
    volatility_dataset = VolatilityDataset("dataset").load("option_SPY_dataset_combined.csv")
    return SSVDataset("dataset").load(volatility_dataset), volatility_dataset


def test(checkpoint):
    viewer = Dataviewer()
    dataset, volatility_dataset = preprocess()
    model = MultiSST(4, 2, 2, 1).load_checkpoint(checkpoint)

    idx = 0
    sample = dataset.sample(idx)

    p, q, r = model.fit_params(sample)
    print(f"p: {p}, q: {q}, r: {r}")

    sabr = ParametricSABR(p, q, r)

    _, S, K, T, rf, div = volatility_dataset.sample(idx)
    true_surface = volatility_dataset.get((-np.inf, np.inf), (-np.inf, np.inf), volatility_dataset.dates[idx])
    viewer.plot(true_surface[['strike', 'maturity', 'iv']])

    K = np.linspace(K.min(), K.max(), 20)
    T = np.linspace(T.min(), T.max(), 20)
    pred_ivol = sabr.smooth_surface(S, K, T, rf, div, beta=0.4)

    viewer.plot_ravel(K, T, pred_ivol)
    viewer.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to save/load models checkpoint', default="weights")

    args = parser.parse_args()

