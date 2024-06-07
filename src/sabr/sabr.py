import warnings

import numpy as np
from scipy.optimize import minimize, curve_fit, OptimizeWarning
from tqdm import tqdm


class SABR:
    @staticmethod
    def forward(t, S, rf, div: float = 0.0):
        return S * np.exp((rf - div) * t)

    @classmethod
    def _ivol(cls, alpha, beta, rho, volvol, F, K, t):
        output = np.zeros_like(K)

        for idx, k in enumerate(K):
            if np.isclose(k, F):
                part_1 = (1.0 - beta) ** 2.0 * alpha ** 2.0 / (24.0 * F ** (2.0 - 2.0 * beta))
                part_2 = rho * beta * alpha * volvol / (4.0 * F ** (1.0 - beta))
                part_3 = (2.0 - 3.0 * rho ** 2) * volvol ** 2.0 / 24.0

                output[idx] = (alpha / F ** (1 - beta)) * (1 + (part_1 + part_2 + part_3) * t)
            else:
                logfK = np.log(F / k)
                fkbpow = (F * k) ** ((1.0 - beta) / 2.0)
                z = volvol * fkbpow * logfK / alpha
                xz = np.log((np.sqrt(np.clip(1.0 - 2.0 * rho * z + z ** 2.0, 0, 1e16)) + z - rho) / (1.0 - rho))

                part_1 = ((1.0 - beta) ** 2.0) * (alpha ** 2.0) / (24.0 * fkbpow ** 2.0)
                part_2 = (rho * beta * volvol * alpha) / (4.0 * fkbpow)
                part_3 = (2.0 - 3.0 * rho ** 2) * volvol ** 2.0 / 24.0
                part_4 = ((1.0 - beta) ** 2) * (logfK ** 2) / 24.0
                part_5 = ((1.0 - beta) ** 4) * (logfK ** 4) / 1920.0

                output[idx] = (alpha * z * (1 + (part_1 + part_2 + part_3) * t)) / (
                        fkbpow * xz * (1 + part_4 + part_5))

        return output

    @classmethod
    def ivol(cls, alpha, beta, rho, volvol, S, K, t, rf, div):
        F = cls.forward(t, S, rf, div)

        return cls._ivol(alpha, beta, rho, volvol, F, K, t)

    @classmethod
    def _calibrate_alpha(cls, beta, rho, volvol, ivol, F, K, t):
        # p_3 = -ivol_atm
        p_3 = -ivol[np.argmin(np.abs(F - K))]
        p_2 = (1 + (2 - 3 * rho ** 2) * volvol ** 2 * t / 24) / F ** (1. - beta)
        p_1 = rho * beta * volvol * t / (4 * F ** (2 - 2 * beta))
        p_0 = (1 - beta) ** 2 * t / (24 * F ** (3 - 3 * beta))

        coeffs = [p_0, p_1, p_2, p_3]

        roots = np.roots(coeffs)

        return roots[(roots.imag == 0) & (roots.real >= 0)].real.min()

    @classmethod
    def calibrate_alpha(cls, beta, rho, volvol, ivol, S, K, t, rf, div):
        F = cls.forward(t, S, rf, div)
        return cls._calibrate_alpha(beta, rho, volvol, ivol, F, K, t)

    @classmethod
    def fit_sabr(cls, ivol, S, K, t, rf, div, beta=None, p0=None):
        def func(k, rho, volvol):
            alpha = cls.calibrate_alpha(beta, rho, volvol, ivol, S, k, t, rf, div)
            return cls.ivol(alpha, beta, rho, volvol, S, k, t, rf, div)

        x = K
        y = ivol

        p0 = p0 if p0 is not None else (0.1, 0.1)

        if beta is None:
            betas = np.linspace(0.1, 0.9, 9)
            prev = np.inf
            best = None
            for b in betas:
                beta = b
                try:
                    res = curve_fit(func, x, y, p0)
                except RuntimeError:
                    continue
                err = np.sum((y - func(x, *res[0])) ** 2)
                if err < prev:
                    prev = err
                    best = b
            beta = best

            with warnings.catch_warnings():
                warnings.simplefilter('always', OptimizeWarning)
                res = curve_fit(func, x, y, p0)

        rho, volvol = res[0]
        alpha = cls.calibrate_alpha(beta, rho, volvol, ivol, S, K, t, rf, div)

        return alpha, beta, rho, volvol


class ParametricSABR:
    def __init__(self, p=None, q=None, r=None):
        self.p = np.zeros(5) if p is None else p
        self.q = np.zeros(4) if q is None else q
        self.r = np.zeros(4) if r is None else r

    def set_params(self, p, q, r):
        self.p = p
        self.q = q
        self.r = r

    def ivol(self, s, k, t, r, d, beta):
        alpha = self.alpha(t, self.p)
        rho = self.rho(t, self.q)
        volvol = self.volvol(t, self.r)

        return SABR.ivol(alpha, beta, rho, volvol, s, k, t, r, d)

    def __call__(self, *args, **kwargs):
        return self.ivol(*args, **kwargs)

    @staticmethod
    def alpha(t, p):  # p = (5,)
        try:
            with np.errstate(invalid='raise', over='raise'):
                return p[0] + p[3] / p[4] * (1 - np.exp(-p[4] * t)) / (p[4] * t) + p[1] / p[2] * np.exp(-p[2] * t)
        except FloatingPointError:
            return 1e-16

    @staticmethod
    def rho(t, q):  # q = (4,)
        return q[0] + q[1] * t + q[2] * np.exp(-q[3] * t)

    @staticmethod
    def volvol(t, r):  # r = (4,)
        return r[0] + r[1] * np.power(np.maximum(t, 1e-16), r[2] + 1e-16) * np.exp(np.clip(r[3] * t + 1e-16, -1e2, 5e2))

    @classmethod
    def fit(cls, func, size, candidates, bounds=None):  # Candidates \mathcal{S} = {(t, param^{t})}
        def error(param):
            return np.sum((np.subtract(func(candidates[:, 0], param),
                                       candidates[:, 1])) ** 2)

        initial_guess = np.random.rand(size)
        res = minimize(error, initial_guess, method='L-BFGS-B', bounds=bounds)

        return res.x

    @classmethod
    def p_alpha(cls, candidates):
        bounds = [(1e-16, None) for _ in range(5)]
        return cls.fit(cls.alpha, 5, candidates, bounds)

    @classmethod
    def p_rho(cls, candidates):
        bounds = [(-1 + 1e-9, 1 - 1e-9), (None, None), (None, None), (0, None)]
        return cls.fit(cls.rho, 4, candidates, bounds)

    @classmethod
    def p_volvol(cls, candidates):
        bounds = [(None, None), (None, None), (None, None), (None, None)]
        return cls.fit(cls.volvol, 4, candidates, bounds)

    @classmethod
    def fit_params(cls, candidates: dict):
        p = cls.p_alpha(candidates["alpha"])
        q = cls.p_rho(candidates["rho"])
        r = cls.p_volvol(candidates["volvol"])

        return p, q, r

    def smooth_surface(self, S, K, T, rf=0.0, div=0.0, beta=0.5):
        iv = np.zeros((len(T), len(K)))

        alpha = self.alpha(T, self.p)
        rho = self.rho(T, self.q)
        volvol = self.volvol(T, self.r)

        for idx, i in tqdm(enumerate(T), total=len(T)):
            iv[idx] = SABR.ivol(alpha[idx], beta, rho[idx], volvol[idx], S, K, i, rf, div)

        return iv  # shape: (len(T), len(K))
