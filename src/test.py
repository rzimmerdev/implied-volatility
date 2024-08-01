import argparse

import numpy as np
import pandas as pd

from src.datasets.dataset_scores import ScoresDataset
from src.datasets.dataset_vol import VolatilityDataset, Dataviewer
from src.models.sst import MultiSST
from src.sabr import SABR


def results(model: MultiSST, score_dataset: ScoresDataset):
    # Save losses and metrics
    for name, sst in zip(("alpha", "rho", "volvol"), (model.z_alpha, model.z_rho, model.z_volvol)):
        pd.DataFrame(sst.losses).to_csv(f"results/{name}_losses.csv", index=False)

    value_datasets = {key: score_dataset.get_dataset(key) for key in ["alpha", "rho", "volvol"]}

    viewer = Dataviewer()

    S, K, T, rf, div, ivol = score_dataset.volatility_dataset.get(0)
    t = T[len(T) // 2]
    _, beta, _, _ = SABR.fit_sabr(ivol[T == t], S, K[T == t], t, rf, div)

    K = np.linspace(K.min(), K.max(), 20)
    T = np.linspace(T.min(), T.max(), 20)

    fig, ax = viewer.create_grid(2, 10)

    for day in range(1, 11):
        S, K_daily, T_daily, rf, div, ivol_daily = score_dataset.volatility_dataset.get(day)

        rows = {key: dataset[day] for key, dataset in value_datasets.items()}
        parametric_sabr, optimal_values = model.get_optimal_parameters(rows, p=0.6)

        ivol_hat = parametric_sabr.smooth_surface(S, K, T, rf, div, beta=beta)

        viewer.plot_ravel(K, T, ivol_hat, ax[0, day - 1])
        viewer.plot(K_daily, T_daily, ivol_daily, ax[1, day - 1])

    viewer.show()


def test():
    scores_dataset = ScoresDataset(VolatilityDataset("dataset"))
    model = MultiSST(4, 4, 8, 8, forward_expansion=256)

    path = "weights"
    model.load_checkpoint(path)

    results(model, scores_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    test()
