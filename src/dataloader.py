import os
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import plotly.offline as pyo
from scipy.interpolate import griddata
from torch.utils.data import Dataset


class VolatilityDataset(Dataset):
    def __init__(self, path="dataset", file="archive"):
        super().__init__()
        self.path = path
        self.file = file

        self.data: pd.DataFrame = pd.DataFrame()

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

    def preprocess(self):
        # divide daysToExpiration by calendar days
        self.data["maturity"] = self.data["daysToExpiration"] / 360

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def get(self, maturity: tuple, strike: tuple, date: str):
        """
        [5 rows x 34 columns]
        Index(['14d', '30d', '3d', '5d', '60d', '7d', 'ask', 'bid', 'bs',
       'daysToExpiration', 'delta', 'dist', 'dist_pct', 'dt', 'expr',
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


class Dataviewer:
    def __init__(self):
        pass

    def view_surface(self, df):
        df = df.dropna().drop_duplicates()
        x = df["strike"].values
        y = df["maturity"].values
        z = df["iv"].values

        # Perform interpolation to fill in missing values
        xi = np.linspace(min(x), max(x), 100)  # Define a grid for x
        yi = np.linspace(min(y), max(y), 100)  # Define a grid for y
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        fig = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi)])
        fig.update_layout(title='IV Surface', autosize=False,
                          width=500, height=500,
                          margin=dict(l=65, r=50, b=65, t=90))
        # add labels
        fig.update_layout(scene=dict(
            xaxis_title='Strike',
            yaxis_title='Maturity',
            zaxis_title='IV'))
        fig.show()


if __name__ == "__main__":
    dataset = VolatilityDataset()
    dataset.load("option_SPY_dataset_combined.csv")
    print(dataset.data.head())
    print(dataset.data.columns)

    data = dataset.get((0.0, 0.2), (300, 400), "2021-01-04")

    viewer = Dataviewer()
    viewer.view_surface(data)
