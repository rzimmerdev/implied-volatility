import os
import zipfile

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import Dataset


class VolatilityDataset(Dataset):
    def __init__(self, path="dataset", file="archive"):
        super().__init__()
        self.path = path
        self.file = file

        self.data: pd.DataFrame = pd.DataFrame()
        self.dates = []

    def load(self, file=None, ticker="SPY"):
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
        # return self.data[
        #     (self.data["maturity"] >= maturity[0]) & (self.data["maturity"] <= maturity[1]) &
        #     (self.data["strike"] >= strike[0]) & (self.data["strike"] <= strike[1])]
        return self.data[
            (self.data["maturity"] >= maturity[0]) & (self.data["maturity"] <= maturity[1]) &
            (self.data["strike"] >= strike[0]) & (self.data["strike"] <= strike[1]) &
            (self.data["dt"] == date)]

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        # useful columns = S (underlying price), K (strike price), T (time to maturity),
        # r (risk free rate), d (dividend rate), IV (implied volatility)
        date = self.dates[idx]

        data = self.data[self.data["dt"] == date]

        return data[["underlying", "strike", "maturity", "r", "d"]].values, data["iv"].values


class Dataviewer:
    def __init__(self):
        matplotlib.use('TkAgg')

    def view_surface(self, df):
        df = df.dropna().drop_duplicates()
        x = df["strike"].values
        y = df["maturity"].values
        z = df["iv"].values

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x, y, z, cmap='viridis')

        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('IV')

        plt.show()


if __name__ == "__main__":
    dataset = VolatilityDataset()
    # https://www.kaggle.com/datasets/shawlu/option-spy-dataset-combinedcsv
    dataset.load("option_SPY_dataset_combined.csv")
    print(dataset.data.head())
    print(dataset.data.columns)

    data = dataset.get((0.0, 0.2), (300, 400), "2021-01-04")

    viewer = Dataviewer()
    viewer.view_surface(data)
