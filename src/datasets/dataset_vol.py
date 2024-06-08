import os
import zipfile

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import Dataset


class VolatilityDataset(Dataset):
    """
    Dataset for volatility data:

    - Input variables are S, K, T, rf, div;
    - Target is implied volatility.
    """
    def __init__(self, path="dataset", file="archive"):
        super().__init__()
        self.path = path
        self.file = file

        self.data: pd.DataFrame = pd.DataFrame()
        self.dates = []

    def load(self, file=None):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        # test if zip is unpacked by checking if
        # the file exists
        if not os.path.exists(f"{self.path}/{file}"):
            with zipfile.ZipFile(f"{self.path}/{self.file}.zip", 'r') as zip_ref:
                zip_ref.extractall(self.path)

        self.data = pd.read_csv(f"{self.path}/{file}")
        self.preprocess()

        start_date = self.data["dt"].min()
        end_date = self.data["dt"].max()

        # risk-free rate
        r = pdr.get_data_fred('DGS10', start=start_date, end=end_date) / 100
        r = r.ffill().reindex(self.data["dt"]).values
        # dividend yield for spy
        d = np.array([1.33 / 100] * len(self.data))

        self.data["r"] = r
        self.data["d"] = d
        return self

    def preprocess(self):
        # divide daysToExpiration by calendar days
        self.data["maturity"] = self.data["daysToExpiration"] / 360
        self.dates = self.data["dt"].unique()

    def get(self, maturity: tuple, strike: tuple, date: str):
        """
        [5 rows x 34 columns]
        Index(['14d', '30d', '3d', '5d', '60d', '7d',
        'ask', 'bid', 'bs', 'daysToExpiration', 'delta', 'dist', 'dist_pct', 'dt', 'expr',
        'extrinsic', 'extrinsic_pct', 'gamma', 'inTheMoney', 'intrinsic',
        'intrinsic_pct', 'iv', 'mark', 'openInterest', 'rho', 'strike',
        'theoreticalOptionValue', 'theoreticalVolatility', 'theta', 'timeValue',
        'underlying', 'vega', 'volatility', 'prev_iv'],
      dtype='object')
        :param maturity: interval of maturity
        :param strike: interval of strike
        :param date: date
        :return:
        """
        return self.data[
            (self.data["maturity"] >= maturity[0]) & (self.data["maturity"] <= maturity[1]) &
            (self.data["strike"] >= strike[0]) & (self.data["strike"] <= strike[1]) &
            (self.data["dt"] == date)]

    def sample(self, idx=0):
        values, target = self[idx]

        return target, values[0, 0], values[:, 1], values[:, 2], values[0, 3], values[0, 4]  # iv, S, K, T, rf, div

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        # columns =
        # S (underlying price),
        # K (strike price),
        # T (time to maturity),
        # rf (risk free rate),
        # div (dividend rate),
        # iv (implied volatility)
        date = self.dates[idx]

        data = self.data[self.data["dt"] == date]

        return data[["underlying", "strike", "maturity", "r", "d"]].values, data["iv"].values


class Dataviewer:
    def __init__(self):
        matplotlib.use('TkAgg')

    @classmethod
    def plot(cls, df):
        k = df["strike"].values
        t = df["maturity"].values
        ivol = df["iv"].values

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_trisurf(k, t, ivol, cmap='viridis')

        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('IV')

    @classmethod
    def plot_ravel(cls, K, T, iv):
        strikes_grid, maturities_grid = np.meshgrid(K, T, indexing='ij')
        df = pd.DataFrame({
            "strike": strikes_grid.ravel(),
            "maturity": maturities_grid.ravel(),
            "iv": iv.ravel()
        })

        cls.plot(df)

    @classmethod
    def show(cls):
        plt.show()
