import argparse

from src.datasets.dataset_ssv import SSVDataset
from src.datasets.dataset_vol import VolatilityDataset
from src.models.sst import MultiSST


def preprocess():
    volatility_dataset = VolatilityDataset("dataset").load("option_SPY_dataset_combined.csv")
    return SSVDataset("dataset").load(volatility_dataset), volatility_dataset


def train(checkpoint):
    dataset, _ = preprocess()
    model = MultiSST(4, 2, 2, 1)

    path = "weights"
    if checkpoint:
        path = checkpoint
    if not model.exists(path):
        model.train(dataset, epochs=100)
        model.save_checkpoint(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--preprocess', action='store_true', help='Preprocess data only')

    args = parser.parse_args()

    if args.preprocess:
        preprocess()
    else:
        parser.add_argument('--checkpoint', type=str, help='Path to save/load models checkpoint', default="weights")
        args = parser.parse_args()

        train(args.checkpoint)
