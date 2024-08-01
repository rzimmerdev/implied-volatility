import argparse

from src.datasets.dataset_scores import ScoresDataset
from src.datasets.dataset_vol import VolatilityDataset
from src.models.sst import MultiSST


def train(checkpoint, batch_size=1, num_workers=8, epochs=100, lr=1e-5):
    scores_dataset = ScoresDataset(VolatilityDataset("dataset"))
    model = MultiSST(4, 4, 8, 8, forward_expansion=256, lr=lr)

    path = "weights"
    if checkpoint:
        path = checkpoint
    if not model.exists(path):
        model.train(scores_dataset, batch_size, num_workers, epochs)
        model.save_checkpoint(path)

        # save train losses to results/{name}_losses.csv [alpha_loss.csv, rho_loss.csv, volvol_loss.csv]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to save/load models checkpoint', default="weights")
    args = parser.parse_args()

    train(args.checkpoint)
