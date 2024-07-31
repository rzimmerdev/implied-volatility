import unittest

import numpy as np

from src.datasets.dataset_vol import VolatilityDataset


class TestVol(unittest.TestCase):
    def test_dataset(self):
        dataset = VolatilityDataset("../dataset")


if __name__ == '__main__':
    unittest.main()
