import matplotlib
from matplotlib import pyplot as plt

from sabr import SABR
from data import VolatilityDataset, plot


def main():
    matplotlib.use("TkAgg")

    data = VolatilityDataset()
    data.load("option_SPY_dataset_combined.csv")

    x, y = data[0]

    sabr = SABR(0.1, 0.5, 0.1, 0.1)
    sabr.fit(x, y)
    y_pred = sabr(x)

    _, ax = plot(x[:, 1], x[:, 2], y)
    plot(x[:, 1], x[:, 2], y_pred, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
