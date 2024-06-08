import unittest

from src.datasets.dataset_vol import VolatilityDataset
from src.models import SSV
from src.sabr import ParametricSABR


class MyTestCase(unittest.TestCase):
    def sample_surface(self, idx=0):
        dataset = VolatilityDataset("../dataset")
        dataset.load("../dataset/option_SPY_dataset_combined.csv")

        return dataset[idx]

    def test_mc(self):
        values, target = self.sample_surface()
        S = values[0, 0]
        K = values[:, 1]
        T = values[:, 2]
        rf = values[0, 3]
        div = values[0, 4]
        ivol = target

        canditates = ParametricSABR.candidates(ivol, S, K, T, rf, div, 0.4)

        alpha_candidates = canditates["alpha"]
        alpha_combinations = SSV.mc_combinations(alpha_candidates[0], alpha_candidates)

        print(alpha_combinations)
        print(alpha_combinations.shape)

    def test_ssv(self):
        values, ivol = self.sample_surface()
        S = values[0, 0]
        K = values[:, 1]
        T = values[:, 2]
        rf = values[0, 3]
        div = values[0, 4]

        alpha_candidates = ParametricSABR.candidates(ivol, S, K, T, rf, div, 0.4)["alpha"]
        z_scores, _ = SSV.z_alpha(alpha_candidates, k=20, p=0.7)

        print(z_scores)


if __name__ == '__main__':
    unittest.main()
