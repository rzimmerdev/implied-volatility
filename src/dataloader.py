import os
import zipfile
import pandas as pd

from torch.utils.data import Dataset


class VolatilityDataset(Dataset):
    def __init__(self, path="dataset", file="archive"):
        super().__init__()
        self.path = path
        self.file = file

        self.data = None

    def load(self, file=None):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        # test if zip is unpacked by checking if
        # the file exists
        if not os.path.exists(f"{self.path}/{file}"):
            with zipfile.ZipFile(f"{self.path}/{self.file}.zip", 'r') as zip_ref:
                zip_ref.extractall(self.path)

        self.data = pd.read_csv(f"{self.path}/{file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # since talking about option volatility surfaces, get all points that have the same maturity
        # and the same strike price, so make permutations of unique values of maturity and strike price
        # strikes = self.data["strike"].unique()
        # maturities = self.data["maturity"].unique()
        return self.data.iloc[idx]


if __name__ == "__main__":
    dataset = VolatilityDataset()
    dataset.load("option_SPY_dataset_combined.csv")
    print(dataset.data.head())
