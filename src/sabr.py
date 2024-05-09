import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


class SABRModel:
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
            try:
                with np.errstate(divide='raise'):
                    return np.sum((cls.ivol_vectorized(i, x) - y) ** 2)
            except FloatingPointError:
                print(i)
                return 1e9

        i0 = np.array(initial_guess)
        bounds = [(1e-16, None), (1e-16, 1), (-1 + 1e-9, 1 - 1e-9), (0, None)]

        res = minimize(error, i0, method='L-BFGS-B', bounds=bounds)

        return SABR(*res.x)


class SABR:
    def __init__(self, alpha, beta, rho, volvol):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.volvol = volvol

    def ivol(self, s, k, t, r, d):
        return SABRModel.ivol(self.alpha, self.beta, self.rho, self.volvol, s, k, t, r, d)

    def ivol_vectorized(self, x):
        return SABRModel.ivol_vectorized([self.alpha, self.beta, self.rho, self.volvol], x)

    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            return self.ivol_vectorized(*args, **kwargs)
        else:
            return self.ivol(*args, **kwargs)


class ParametricSABR:
    def __init__(self, prev_tenor: dict):
        self.raw_p = np.random.rand(5)
        self.raw_q = np.random.rand(3)
        self.raw_r = np.random.rand(4)

        self.prev_tenors = prev_tenor
        self.corrected_p = np.zeros(5)
        self.corrected_q = np.zeros(3)
        self.corrected_r = np.zeros(4)

    def p(self, t, candidates=None):
        p = self.raw_p if candidates is None else candidates
        return p[0] + p[3] / p[4] * (1 - np.exp(-p[4] * t)) / (p[4] * t) + p[1] / p[2] * np.exp(-p[2] * t)

    def q(self, t, candidates=None):
        q = self.raw_q if candidates is None else candidates
        return q[0] + q[1] * t + q[2] * np.exp(-q[3] * t)

    def r(self, t, candidates=None):
        r = self.raw_r if candidates is None else candidates
        return r[0] + r[1] * np.power(t, r[2]) * np.exp(r[3] * t)

    def corrected_params(self) -> list:  # shape: (prev_tenor_size, 3)
        params = []
        for tenor in self.prev_tenors.keys():
            tenor = float(tenor)
            params.append(
                (self.p(tenor, self.corrected_p), self.q(tenor, self.corrected_q), self.r(tenor, self.corrected_r))
            )

        return params

    @staticmethod
    def param_star(func, size, t, candidates):
        def error(p, candidate):
            # p => 5,
            # candidate => 5,n
            return np.sum((func(t, p) - candidate) ** 2)

        initial_guess = np.random.rand(size)
        res = minimize(error, initial_guess, args=(candidates,), method='L-BFGS-B')

        return res.x

    def p_star(self, t, candidates):
        return self.param_star(self.p, 5, t, candidates)

    def q_star(self, t, candidates):
        return self.param_star(self.q, 3, t, candidates)

    def r_star(self, t, candidates):
        return self.param_star(self.r, 4, t, candidates)

    def smooth_surface(self, K, T, star_params = None):

        if star_params is None:
            p = self.p_

        for k in K:
            for t in T:
                alpha =


def main():
    s = 100
    k = np.linspace(80, 120, 100)
    t = [0.1, 0.5, 1, 2, 3, 5, 7, 10]
    r = 0.05
    d = 0.02

    sabrs = []

    for i in range(len(t)):
        x = np.array([[s, k, t[i], r, d] for k in k])
        y = SABRModel.ivol(0.1, 0.5, 0.1, 0.1, s, k, t[i], r, d)
        sabr = SABRModel.fit([0.1, 0.5, 0.1, 0.1], x, y)
        sabrs.append(sabr)

    for i, sabr in enumerate(sabrs):
        iv = np.array([sabr.ivol(s, strike, t[i], r, d) for strike in k])
        plt.plot(k, iv, label=f"{t[i]}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
