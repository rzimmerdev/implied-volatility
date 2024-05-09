import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from sabr import SABRModel
from data import VolatilityDataset, plot

import torch
from torch import optim, nn
from src.transformer import Transformer


def main():
    data = VolatilityDataset()
    data.load("option_SPY_dataset_combined.csv")

    def test_sabr():
        matplotlib.use("TkAgg")

        x_train, y_train = data[0]
        plot(x_train[:, 1], x_train[:, 2], y_train)
        plt.show()

        maturities = x_train[:, 2]
        unique_t = np.unique(maturities)

        sabrs = []

        for t in unique_t:
            x = x_train[x_train[:, 2] == t]
            y = y_train[x_train[:, 2] == t]

            sabr = SABRModel.fit([0.1, 0.5, 0.1, 0.1], x, y)
            sabrs.append(sabr)

        for i, sabr in enumerate(sabrs):
            x = x_train[x_train[:, 2] == unique_t[i]]
            iv = sabr.ivol_vectorized(x)

            plt.plot(x[:, 1], iv, label=f"{unique_t[i]}")

        plt.legend()
        plt.show()

    def test_transformer():
        dataloader = DataLoader(data, batch_size=64, shuffle=True)
        in_features = len(data[0][0][0])
        heads = 1
        num_layers = 4
        out_features = 1
        model = Transformer(in_features, heads, num_layers, out_features)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-6)

        for epoch in range(10):
            for x, y in dataloader:
                print(x.shape)
                optimizer.zero_grad()
                output = model(*x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                print(loss.item())

    test_transformer()


if __name__ == "__main__":
    main()
