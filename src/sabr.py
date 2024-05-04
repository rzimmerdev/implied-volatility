import math
import numpy as np

import matplotlib.pyplot as plt


class SABR:
    def __init__(self, a, b, r):
        self.a = a
        self.b = b
        self.r = r

    def f(self, S, K, T):
        # Hagan et al. (2002) forward price
        return S - self.a / self.b * (S * K) ** ((1 - self.b) / 2) * math.log(S / K)

    def z(self, S, K, T):
        # Hagan et al. (2002) z
        return self.a / self.b * (S * K) ** ((1 - self.b) / 2) * math.log(S / K)

    def ivol(self, S, K, T):
        # Hagan et al. (2002) implied volatility
        z = self.z(S, K, T)
        X = math.log((math.sqrt(1 - 2 * self.r * z + z ** 2) + z - self.r) / (1 - self.r))
        return self.a / (S * K) ** ((1 - self.b) / 2) * (1 + (1 - self.b) ** 2 / 24 * z ** 2 + (1 - self.b) ** 4 / 1920 * z ** 4) * T

    def preview(self, ax):
        S = 100
        T = np.linspace(0.1, 2, 10)
        K = np.array([
            np.random.randint(80, 120, 100) for _ in T
        ])

        IV = np.array([
            [self.ivol(S, k, t) for k in K[i]] for i, t in enumerate(T)
        ])

        # flatten K and IV, and make T the same size
        T = np.array([
            [t] * len(K[i]) for i, t in enumerate(T)
        ]).flatten()

        K = K.flatten()
        IV = IV.flatten()

        return S, K, T, IV


def main():
    # draw simple volatility smile
    import matplotlib

    matplotlib.use("TkAgg")

    A = np.arange(0.01, 0.5, 0.1)  # len = 5
    B = np.arange(0.1, 0.6, 0.1)  # len = 4
    r = 0.1
    # total graphs = 5 * 4 * 1 = 20

    fig = plt.figure(figsize=(8, 8))
    plot_index = 1

    for i, a in enumerate(A):
        for j, b in enumerate(B):
            # Create a new subplot for each combination of A and B
            ax = fig.add_subplot(len(A), len(B), plot_index, projection='3d')
            ax.set_title(f"a={a:.1f}, b={b:.1f}, r={r:.1f}")
            sabr = SABR(a, b, r)
            # ignore pycharm warning
            # noinspection PyTypeChecker
            _, K, T, IV = sabr.preview(ax)
            ax.plot_trisurf(K, T, IV, cmap='viridis')
            plot_index += 1

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()


if __name__ == '__main__':
    main()
