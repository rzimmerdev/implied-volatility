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
    def fit(cls, x, y, initial_guess = None):
        def error(i):
            try:
                with np.errstate(divide='raise'):
                    return np.sum((cls.ivol_vectorized(i, x) - y) ** 2)
            except FloatingPointError:
                print(i)
                return 1e9

        if initial_guess is None:
            initial_guess = [0.1, 0.5, 0.1, 0.1]
        i0 = np.array(initial_guess)
        bounds = [(1e-16, None), (1e-16, 1), (-1 + 1e-9, 1 - 1e-9), (0, None)]

        res = minimize(error, i0, method='L-BFGS-B', bounds=bounds)

        return res.x


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
    def __init__(self, rf, div, prev_tenor: dict = None):
        self.corrected_p = np.zeros(5)
        self.corrected_q = np.zeros(3)
        self.corrected_r = np.zeros(4)

        self.rf = rf
        self.div = div

    def alpha(self, t, p):
        try:
            with np.errstate(invalid='raise', over='raise'):
                return p[0] + p[3] / p[4] * (1 - np.exp(-p[4] * t)) / (p[4] * t) + p[1] / p[2] * np.exp(-p[2] * t)
        except FloatingPointError:
            return 1e-16

    def rho(self, t, q):
        return q[0] + q[1] * t + q[2] * np.exp(-q[3] * t)

    def volvol(self, t, r):
        return r[0] + r[1] * np.power(t, r[2]) * np.exp(r[3] * t)

    @staticmethod
    def param_star(func, size, candidates):  # Candidates \mathcal{S} = {(t, param^{t})}
        def error(param):
            try:
                with np.errstate(invalid='raise', over='raise'):
                    err = np.sum((np.subtract(func(candidates[:, 0], param), candidates[:, 1])) ** 2)
            except FloatingPointError:
                err = 1e9
            return err

        initial_guess = np.random.rand(size)
        res = minimize(error, initial_guess, method='L-BFGS-B')

        return res.x

    def p_star(self, candidates):
        return self.param_star(self.alpha, 5, candidates)

    def q_star(self, candidates):
        return self.param_star(self.rho, 4, candidates)

    def r_star(self, candidates):
        return self.param_star(self.volvol, 4, candidates)

    def smooth_surface(self, K, T, star_params):
        beta = 0.5

        iv = np.zeros((len(T), len(K)))

        for idx, i in enumerate(T):
            alpha = self.alpha(i, star_params["p"])
            rho = self.rho(i, star_params["q"])
            volvol = self.volvol(i, star_params["r"])

            iv[idx] = SABRModel.ivol(alpha, beta, rho, volvol, 100, K, i, self.rf, self.div)  # shape: (len(K),)

        return iv  # shape: (len(T), len(K))


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
        sabr = SABR(*SABRModel.fit([0.1, 0.5, 0.1, 0.1], x, y))
        sabrs.append(sabr)

    for i, sabr in enumerate(sabrs):
        iv = np.array([sabr.ivol(s, strike, t[i], r, d) for strike in k])
        plt.plot(k, iv, label=f"{t[i]}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
