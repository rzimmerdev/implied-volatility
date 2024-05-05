from sabr import SABR
from data import VolatilityDataset


def main():
    data = VolatilityDataset()

    sabr = SABR(0.1, 0.5, 0.1, 0.1)

    # next(iter(data)) --> (x, y)
    # fit receives ([x...
    sabr.fit(iter(data))


if __name__ == "__main__":
    main()
