import unittest

from src.datasets import SSVDataset, ParamDataset


class MyTestCase(unittest.TestCase):
    def test_preprocess(self):
        dataset = SSVDataset("../dataset/").load()
        print(dataset.data)

    def test_dataset(self):
        dataset = ParamDataset(SSVDataset("../dataset/").load(), "alpha")
        print(dataset[0])


if __name__ == '__main__':
    unittest.main()
