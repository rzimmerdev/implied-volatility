from typing import Tuple

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt


class Helper:
    @staticmethod
    def forward(s, t, r: float, d: float = 0):
        return s * np.exp(r * t - d)

    @staticmethod
    def z(alpha, beta, volvol, f, k, t, rho):
        return volvol / alpha * (f * k) ** ((1 - beta) / 2) * np.log(f / k)

    @staticmethod
    def x(alpha, beta, volvol, f, k, rho):
        return np.log((np.sqrt(1 - 2 * rho *
                               Helper.z(alpha, beta, volvol, f, k, 0, rho) +
                               Helper.z(alpha, beta, volvol, f, k, 0, rho) ** 2) +
                       Helper.z(alpha, beta, volvol, f, k, 0, rho) - rho) / (1 - rho))

    @staticmethod
    def ivol(alpha, beta, rho, volvol, s, k, t, r, d=None):
        if d is None:
            d = [0] * len(t)
        f = Helper.forward(s, t, r, d)
        return alpha * (f * k) ** ((1 - beta) / 2) * (1 + (Helper.x(alpha, beta, volvol, f, k, rho) ** 2) / 24 +
                                                      (Helper.x(alpha, beta, volvol, f, k, rho) ** 4) / 1920)

    @staticmethod
    def ivol_vectorized(params, x):
        alpha, beta, rho, volvol = params
        s, k, t, r, d = x.T
        if d is None:
            d = [0] * len(t)
        f = Helper.forward(s, t, r, d)
        x = Helper.x(alpha, beta, volvol, f, k, rho)
        return alpha * (f * k) ** ((1 - beta) / 2) * (1 + (x ** 2) / 24 + (x ** 4) / 1920)

class SABR:
    def __init__(self, alpha, beta, rho, volvol: float = 0):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.volvol = volvol

    def fit(self, x, y):
        def error(i):
            with np.errstate(divide='raise'):
                return np.sum((Helper.ivol_vectorized(i, x) - y) ** 2)
        i0 = np.array([self.alpha, self.beta, self.rho, self.volvol])
        bounds = [(1e-16, None), (1e-16, 1), (-1 + 1e-9, 1 - 1e-9), (0, None)]

        res = minimize(error, i0, method='L-BFGS-B', bounds=bounds)

        self.alpha, self.beta, self.rho, self.volvol = res.x
        return self.alpha, self.beta, self.rho, self.volvol

    def ivol(self, s, k, t, r, d=0):
        return Helper.ivol(self.alpha, self.beta, self.rho, self.volvol, s, k, t, r, d)

    def __call__(self, x):
        s, k, t, r, d = x.T
        return self.ivol(s, k, t, r, d)

    def preview(self, x) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        if x is None:
            s = 100
            t = np.linspace(0.1, 2, 10)
            k = np.array([
                np.linspace(80, 120, 100) for _ in t
            ])
            r = np.array([0.14] * len(t))
            d = np.array([0.1] * len(t))
        else:
            s, k, t, r, d = x.T

        ivol = np.array([
            [self.ivol(s, strike, maturity, r) for strike in k[i]] for i, maturity in enumerate(t)
        ])

        # flatten K and IV, and make T the same size
        t = np.array([
            [maturity] * len(k[i]) for i, maturity in enumerate(t)
        ]).flatten()

        k = k.flatten()
        ivol = ivol.flatten()

        return s, k, t, ivol


def plot_3d(A, B, r):
    # total graphs = 5 * 4 * 1 = 20

    fig = plt.figure(figsize=(12, 12))
    plot_index = 1

    for i, a in enumerate(A):
        for j, b in enumerate(B):
            # Create a new subplot for each combination of A and B
            ax = fig.add_subplot(len(A), len(B), plot_index, projection='3d')
            ax.set_title(f"a={a:.1f}, b={b:.1f}, r={r:.1f}")
            sabr = SABR(a, b, r, 0.1)
            # ignore pycharm warning
            # noinspection PyTypeChecker
            _, K, T, IV = sabr.preview()
            ax.plot_trisurf(K, T, IV, cmap='viridis')
            plot_index += 1

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # change subplots size
    fig.set_size_inches(18.5, 16.5)
    plt.show()


def plot_2d(a, b, r):
    sabr = SABR(a, b, r, 0.1)
    _, K, T, IV = sabr.preview()

    # plot 2d cuts of the 3d surface
    # choose h and w to be closer to a square

    unique_T = np.unique(T)

    total_size = len(np.unique(T))
    h = total_size // 2
    w = 2
    fig, ax = plt.subplots(h, w)

    for i, t in enumerate(unique_T):
        x = K[T == t]
        y = IV[T == t]

        idx, idy = i // 2, i % 2
        ax[idx, idy].plot(x, y)
        ax[idx, idy].set_title(f"T={t:.1f}")
        ax[idx, idy].set_xlabel('Strike')
        ax[idx, idy].set_ylabel('IV')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # change subplots size
    fig.set_size_inches(18.5, 16.5)
    plt.show()


def main():
    # draw simple volatility smile
    def test_plot():
        import matplotlib

        matplotlib.use("TkAgg")

        A = np.arange(0.01, 0.4, 0.1)  # len = 5
        B = np.arange(0.1, 0.5, 0.1)  # len = 4
        r = 0.1

        plot_2d(0.1, 0.1, 0.1)
        plot_3d(A, B, r)

    def test_fit():
        x = np.array([
            [100, 100, 0.1, 0.1],
            [100, 100, 0.2, 0.1],
            [100, 100, 0.3, 0.1],
            [100, 100, 0.4, 0.1],
            [100, 100, 0.5, 0.1],
            [100, 100, 0.6, 0.1],
            [100, 100, 0.7, 0.1],
            [100, 100, 0.8, 0.1],
            [100, 100, 0.9, 0.1],
            [100, 100, 1.0, 0.1],
        ])

        y = np.array([
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        ])

        sabr = SABR(0.1, 0.5, 0.1, 0.1)
        print(sabr.fit(x, y))

    test_fit()


if __name__ == '__main__':
    main()
