import unittest

import numpy as np

from src.datasets.dataset_scores import ScoresDataset, SSV
from src.datasets.dataset_vol import VolatilityDataset


class TestScores(unittest.TestCase):
    def test_ssv(self):
        key = "alpha"
        scoring_function = SSV.get_funcs(key, k=10, p=0.8, sigma=1)

        candidates = np.random.rand(20, 2)
        scores, param_star = scoring_function(candidates)

    def test_preprocess(self):
        dataset = ScoresDataset(VolatilityDataset("../dataset"))
        alpha_dataset = dataset.get_dataset("alpha")
        x, y = next(iter(alpha_dataset))

        print(x.shape, y.shape)


if __name__ == '__main__':
    unittest.main()
