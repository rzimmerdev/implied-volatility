import argparse

import numpy as np

from src.datasets.dataset_ssv import SSVDataset
from src.datasets.dataset_vol import VolatilityDataset, Dataviewer
from src.models.sst import MultiSST
from src.sabr import ParametricSABR


def train(checkpoint):
    volatility_dataset = VolatilityDataset("dataset").load("option_SPY_dataset_combined.csv")
    dataset = SSVDataset("dataset").load(volatility_dataset)
    model = MultiSST(4, 2, 2, 1)

    path = "weights"
    if checkpoint:
        path = checkpoint
    if not model.exists(path):
        model.train(dataset, epochs=100)
        model.save_checkpoint(path)


def test(checkpoint):
    viewer = Dataviewer()
    volatility_dataset = VolatilityDataset("dataset").load("option_SPY_dataset_combined.csv")
    dataset = SSVDataset("dataset").load(volatility_dataset)
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
    # args
    parser = argparse.ArgumentParser()
    # whether or not to load a checkpoint
    parser.add_argument('--checkpoint', type=str, help='Path to save/load models checkpoint', default="weights")

    args = parser.parse_args()

    train(args.checkpoint)
    test(args.checkpoint)
