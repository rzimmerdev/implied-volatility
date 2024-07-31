import warnings

import numpy as np
from scipy.optimize import minimize, OptimizeWarning
from .sabr import SABR


class ParametricSABR:
    def __init__(self, p=None, q=None, r=None):
        self.p = np.zeros(5) if p is None else p
        self.q = np.zeros(4) if q is None else q
        self.r = np.zeros(4) if r is None else r

    def ivol(self, s, k, t, r, d, beta):
        alpha = self.alpha(t, self.p)
        rho = self.rho(t, self.q)
        volvol = self.volvol(t, self.r)

        return SABR.ivol(alpha, beta, rho, volvol, s, k, t, r, d)

    def __call__(self, *args, **kwargs):
        return self.ivol(*args, **kwargs)

    @staticmethod
    def alpha(t, p):  # p = (5,)
        return p[0] + p[3] / p[4] * (1 - np.exp(-p[4] * t)) / (p[4] * t) + p[1] / p[2] * np.exp(-p[2] * t)

    @staticmethod
    def rho(t, q):
        return q[0] + q[1] * t + q[2] * np.exp(-q[3] * t)

    @staticmethod
    def volvol(t, r):
        return r[0] + r[1] * np.power(t, r[2]) * np.exp(r[3] * t)

    @classmethod
    def funcs(cls):
        return {
            "alpha": cls.alpha,
            "rho": cls.rho,
            "volvol": cls.volvol
        }

    @classmethod
    def fit(cls, func, size, candidates, constraints=None):  # Candidates \mathcal{S} = {(t, param^{t})}
        def error(param):
            return np.sum((np.subtract(func(candidates[:, 0], param), candidates[:, 1])) ** 2)

        initial_guess = np.random.rand(size)
        initial_guess = 0.5 + (initial_guess - initial_guess.mean()) / initial_guess.std()

        # ignore warnings, so as to not print to console
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(error, initial_guess, constraints=constraints)

        return res.x

    @classmethod
    def fit_p(cls, candidates):
        def constraint(param):
            return cls.alpha(candidates[:, 0], param)

        constraints = [{'type': 'ineq', 'fun': constraint}]
        return cls.fit(cls.alpha, 5, candidates, constraints=constraints)

    @classmethod
    def fit_q(cls, candidates):
        # func(candidates[:, 0], param) = rho, should be between -1, 1
        def lower_bound(param):
            return -1 - cls.rho(candidates[:, 0], param)

        def upper_bound(param):
            return cls.rho(candidates[:, 0], param) - 1

        constraints = [{'type': 'ineq', 'fun': lower_bound}, {'type': 'ineq', 'fun': upper_bound}]

        return cls.fit(cls.rho, 4, candidates, constraints)

    @classmethod
    def fit_r(cls, candidates):
        # func(candidates[:, 0], param) = volvol, should be between 0, np.inf
        def constraint(param):
            return cls.volvol(candidates[:, 0], param)

        constraints = [{'type': 'ineq', 'fun': constraint}]

        return cls.fit(cls.volvol, 4, candidates, constraints)

    @classmethod
    def fit_params(cls, candidates: dict):
        return cls.fit_p(candidates["alpha"]), cls.fit_q(candidates["rho"]), cls.fit_r(candidates["volvol"])

    def smooth_surface(self, S, K, T, rf=0.0, div=0.0, beta=0.5):
        if self.p is None or self.q is None or self.r is None:
            raise ValueError("Parameters not set")

        iv = np.zeros((len(T), len(K)))

        alpha = self.alpha(T, self.p)
        rho = self.rho(T, self.q)
        volvol = self.volvol(T, self.r)

        for idx, i in enumerate(T):
            iv[idx] = SABR.ivol(alpha[idx], beta, rho[idx], volvol[idx], S, K, i, rf, div)

        return iv  # shape: (len(T), len(K))
