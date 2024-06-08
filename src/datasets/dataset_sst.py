import torch
from torch.utils.data import Dataset

from .dataset_ssv import SSVDataset


class ParamDataset(Dataset):
    def __init__(self, ssv_dataset: SSVDataset, param):
        self.ssv_dataset = ssv_dataset
        self.param = param
        self.dates = self.ssv_dataset.data[param]["date"].unique()

    def load(self):
        self.ssv_dataset.load()
        return self

    def __len__(self):
        # get number of unique dates
        return len(self.dates)

    def __getitem__(self, idx):
        # get all with same idx
        rows = self.ssv_dataset.data[self.param][self.ssv_dataset.data[self.param]["date"] == self.dates[idx]]

        # input is 1:4, output is 5
        x = rows.iloc[:, 1:5].values
        y = rows.iloc[:, 5].values

        # convert to torch
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    @classmethod
    def map(cls, rows):


        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
