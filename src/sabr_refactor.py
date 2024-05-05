import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


class SABR:
    @classmethod
    def forward(cls, t, s, r, d: float = 0.0):
        return s * np.exp((r - d) * t)

    @classmethod
    def z(cls, volvol, alpha, beta, f, k):
        return volvol / alpha * (f * k) ** ((1 - beta) / 2) * np.log(f / k)

    @classmethod
    def x(cls, volvol, alpha, beta, f, k, rho):
        return np.log((np.sqrt(1 - 2 * rho * cls.z(volvol, alpha, beta, f, k) + cls.z(volvol, alpha, beta, f, k) ** 2) +
                       cls.z(volvol, alpha, beta, f, k) - rho) / (1 - rho))

    @classmethod
    def ivol(cls, alpha, beta, rho, volvol, s, k, t, r, d: float = 0.0):
        f = cls.forward(t, s, r, d)
        return alpha * (f * k) ** ((1 - beta) / 2) * (1 + (cls.x(volvol, alpha, beta, f, k, rho) ** 2) / 24 +
                                                      (cls.x(volvol, alpha, beta, f, k, rho) ** 4) / 1920)

    @classmethod
    def ivol_vectorized(cls, i, x):
        alpha, beta, rho, volvol = i
        s, k, t, r, d = x.T

        f = cls.forward(t, s, r, d)
        x = cls.x(volvol, alpha, beta, f, k, rho)
        return alpha * (f * k) ** ((1 - beta) / 2) * (1 + (x ** 2) / 24 + (x ** 4) / 1920)

    @classmethod
    def fit(cls, initial_guess, x, y):
        def error(i):
            with np.errstate(divide='raise'):
                return np.sum((cls.ivol_vectorized(i, x) - y) ** 2)

        i0 = np.array(initial_guess)
        bounds = [(1e-16, None), (1e-16, 1), (-1 + 1e-9, 1 - 1e-9), (0, None)]

        res = minimize(error, i0, method='L-BFGS-B', bounds=bounds)

        return SABRModel(*res.x)


class SABRModel:
    def __init__(self, alpha, beta, rho, volvol):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.volvol = volvol

    def ivol(self, s, k, t, r, d):
        return SABR.ivol(self.alpha, self.beta, self.rho, self.volvol, s, k, t, r, d)

    def ivol_vectorized(self, x):
        return SABR.ivol_vectorized([self.alpha, self.beta, self.rho, self.volvol], x)


def main():
    s = 100
    k = np.linspace(80, 120, 100)
    t = [0.1, 0.5, 1, 2, 3, 5, 7, 10]
    r = 0.05
    d = 0.02

    sabrs = []

    for i in range(len(t)):
        x = np.array([[s, k, t[i], r, d] for k in k])
        y = SABR.ivol(0.1, 0.5, 0.1, 0.1, s, k, t[i], r, d)
        sabr = SABR.fit([0.1, 0.5, 0.1, 0.1], x, y)
        sabrs.append(sabr)

    for i, sabr in enumerate(sabrs):
        iv = np.array([sabr.ivol(s, strike, t[i], r, d) for strike in k])
        plt.plot(k, iv, label=f"{t[i]}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
