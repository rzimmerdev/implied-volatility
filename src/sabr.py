import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt


class SABR:
    def __init__(self, alpha, beta, rho, volvol: float = 0):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.volvol = volvol


    @staticmethod
    def forward(s, t, r: float, d: float = 0):
        return s * np.exp(r * t - d)

    def z(self, f, k, t):
        return self.volvol / self.alpha * (f * k) ** ((1 - self.beta) / 2) * np.log(f / k)

    def x(self, f, k):
        return np.log((np.sqrt(1 - 2 * self.rho * self.z(f, k, 0) + self.z(f, k, 0) ** 2) + self.z(f, k, 0) - self.rho) / (1 - self.rho))

    def ivol(self, s, k, t, r, d=0):
        f = self.forward(s, t, r, d)
        return self.alpha * (f * k) ** ((1 - self.beta) / 2) * (1 + (self.x(f, k) ** 2) / 24 + (self.x(f, k) ** 4) / 1920)

    def fit(self, s, k, t, iv):
        """
        :param s: spot price
        :param k: strike
        :param t: time to maturity
        :param iv: implied volatility
        :return: alpha, beta, rho
        """
        def error(x):
            return np.sum((self.ivol(s, k, t, 0, 0) - iv) ** 2)

        x0 = np.array([self.alpha, self.beta, self.rho])
        bounds = [(0, None), (0, 1), (-1, 1)]

        res = minimize(error, x0, method='L-BFGS-B', bounds=bounds)
        self.alpha, self.beta, self.rho = res.x

        return self.alpha, self.beta, self.rho

    def preview(self, ax, r, d=0):
        s = 100
        t = np.linspace(0.1, 2, 10)
        k = np.array([
            np.random.randint(80, 120, 100) for _ in t
        ])

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


def main():
    # draw simple volatility smile
    import matplotlib

    matplotlib.use("TkAgg")

    A = np.arange(0.01, 0.5, 0.1)  # len = 5
    B = np.arange(0.1, 0.5, 0.1)  # len = 4
    r = 0.1
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
            _, K, T, IV = sabr.preview(ax, r)
            ax.plot_trisurf(K, T, IV, cmap='viridis')
            plot_index += 1

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # change subplots size
    fig.set_size_inches(18.5, 16.5)
    plt.show()


if __name__ == '__main__':
    main()
