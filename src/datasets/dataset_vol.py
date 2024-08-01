import os
import zipfile

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt


class VolatilityDataset:
    def __init__(self, path="dataset", file="archive.zip"):
        super().__init__()
        self.path = path
        self.file = file

        self.data: pd.DataFrame = pd.DataFrame()
        self.dates = []
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if not os.path.exists(f"{self.path}/{self.file}"):
            with zipfile.ZipFile(f"{self.path}/{self.file}", 'r') as zip_ref:
                zip_ref.extractall(self.path)

        self.data = pd.read_csv(f"{self.path}/{self.file}")

        self.data["maturity"] = self.data["daysToExpiration"] / 360
        self.dates = self.data["dt"].unique()

        start_date = self.data["dt"].min()
        end_date = self.data["dt"].max()

        r = pdr.get_data_fred('DGS10', start=start_date, end=end_date) / 100
        r = r.ffill().reindex(self.data["dt"]).values
        d = np.array([1.33 / 100] * len(self.data))

        self.data["r"] = r
        self.data["d"] = d
        return self

    def __len__(self):
        return len(self.dates)

    def get(self, idx):
        date = self.dates[idx]
        data = self.data[(self.data["dt"] == date)]
        values = data[["underlying", "strike", "maturity", "r", "d"]].values
        target = data["iv"].values
        return values[0, 0], values[:, 1], values[:, 2], values[0, 3], values[0, 4], target


class Dataviewer:
    def __init__(self):
        import matplotlib
        matplotlib.use('TkAgg')

    @classmethod
    def plot_df(cls, df, ax=None):
        k = df["strike"].values
        t = df["maturity"].values
        ivol = df["iv"].values

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        ax.plot_trisurf(k, t, ivol, cmap='viridis')

        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('IV')

    @classmethod
    def plot_list(cls, K, T, iv, ax=None):
        df = pd.DataFrame({
            "strike": np.concatenate(K),
            "maturity": np.concatenate(T),
            "iv": np.concatenate(iv)
        })

        cls.plot_df(df, ax)

    @classmethod
    def plot_smile(cls, K, ivols, ax=None, t=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(K, ivols, label="SABR Smile")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied Volatility")
        ax.set_title(f"Implied Volatility Smile - Maturity: {t}")
        ax.legend()

        return ax

    @classmethod
    def plot(cls, K, T, iv, ax=None):
        df = pd.DataFrame({
            "strike": K,
            "maturity": T,
            "iv": iv
        })

        cls.plot_df(df, ax)

    @classmethod
    def plot_ravel(cls, K, T, iv, ax=None):
        strikes_grid, maturities_grid = np.meshgrid(K, T, indexing='ij')
        df = pd.DataFrame({
            "strike": strikes_grid.ravel(),
            "maturity": maturities_grid.ravel(),
            "iv": iv.ravel()
        })

        cls.plot_df(df, ax)

    @classmethod
    def show(cls):
        plt.show()

    @classmethod
    def create_grid(cls, rows, columns, fig_size=(30, 20)):
        fig, ax = plt.subplots(rows, columns, subplot_kw={'projection': '3d'}, figsize=fig_size)
        return fig, ax