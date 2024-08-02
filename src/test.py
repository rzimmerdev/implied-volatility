import argparse

import numpy as np
from matplotlib import pyplot as plt

from src.datasets.dataset_scores import ScoresDataset
from src.datasets.dataset_vol import VolatilityDataset, Dataviewer
from src.models.sst import MultiSST
from src.sabr import SABR


def results(model: MultiSST, score_dataset: ScoresDataset):
    value_datasets = {key: score_dataset.get_dataset(key) for key in ["alpha", "rho", "volvol"]}

    viewer = Dataviewer()

    S, K, T, rf, div, ivol = score_dataset.volatility_dataset.get(0)
    t = T[len(T) // 2]
    _, beta, _, _ = SABR.fit_sabr(ivol[T == t], S, K[T == t], t, rf, div)

    K = np.linspace(K.min(), K.max(), 20)
    T = np.linspace(T.min(), T.max(), 20)

    k = 3
    fig, ax = viewer.create_grid(2, k)
    idx = np.random.randint(1, len(score_dataset) - k)

    for day in range(idx, idx + k):
        S, K_daily, T_daily, rf, div, ivol_daily = score_dataset.volatility_dataset.get(day)

        rows = {key: dataset[day] for key, dataset in value_datasets.items()}
        parametric_sabr, optimal_values = model.get_optimal_parameters(rows, p=0.6)

        ivol_hat = parametric_sabr.smooth_surface(S, K, T, rf, div, beta=beta)

        viewer.plot_ravel(K, T, ivol_hat, ax[0, day - idx])

        # from daily, remove points that are outliers
        std = ivol_daily.std()
        mean = ivol_daily.mean()
        pos = []
        for i, ivol in enumerate(ivol_daily):
            if ivol > mean + 2 * std or ivol < mean - 2 * std:
                pos.append(i)

        K_daily = np.delete(K_daily, pos)
        T_daily = np.delete(T_daily, pos)
        ivol_daily = np.delete(ivol_daily, pos)

        viewer.plot(K_daily, T_daily, ivol_daily, ax[1, day - idx])

    viewer.show()


def test():
    scores_dataset = ScoresDataset(VolatilityDataset("dataset"))
    model = MultiSST(4, 4, 8, 4, forward_expansion=8, fc_size=256)

    path = "weights"
    model.load_checkpoint(path)

    # plot train losses (results/{model}_losses.csv)
    losses = {key: np.loadtxt(f"results/{key}_losses.csv", delimiter=",") for key in ["alpha", "rho", "volvol"]}
    max_size = min([len(loss) for loss in losses.values()])
    for name in ["alpha", "rho", "volvol"]:
        window = 100
        current_loss = losses[name][:max_size]
        moving_average = np.convolve(current_loss, np.ones(window) / window, mode='valid')
        plt.plot(moving_average, label=name)

    plt.legend()
    plt.show()

    results(model, scores_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    test()
