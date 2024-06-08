import warnings

import numpy as np
from scipy.optimize import curve_fit


class SABR:
    @staticmethod
    def forward(t, S, rf, div: float = 0.0):
        return S * np.exp((rf - div) * t)

    @classmethod
    def _ivol(cls, alpha, beta, rho, volvol, F, K, t):
        logfK = np.log(F / K)
        fkbpow = (F * K) ** ((1.0 - beta) / 2.0)
        z = volvol * fkbpow * logfK / alpha

        # if invalid value encountered in log, return 0
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                xz = np.log((np.sqrt(np.clip(1.0 - 2.0 * rho * z + z ** 2.0, 0, 1e16)) + z - rho) / (1.0 - rho))
            except RuntimeWarning:
                return 1e-16

        part_1 = ((1.0 - beta) ** 2.0) * (alpha ** 2.0) / (24.0 * fkbpow ** 2.0)
        part_2 = (rho * beta * volvol * alpha) / (4.0 * fkbpow)
        part_3 = (2.0 - 3.0 * rho ** 2) * volvol ** 2.0 / 24.0
        part_4 = ((1.0 - beta) ** 2) * (logfK ** 2) / 24.0
        part_5 = ((1.0 - beta) ** 4) * (logfK ** 4) / 1920.0

        output = (alpha * z * (1 + (part_1 + part_2 + part_3) * t)) / (
                fkbpow * xz * (1 + part_4 + part_5) + 1e-16)

        nans = np.isnan(output)

        try:
            if np.any(nans):
                output[nans] = np.interp(K[nans], K[~nans], output[~nans])
        except ValueError:
            output = np.clip(output, 0, 1e16)

        infs = np.isinf(output)

        try:
            if np.any(infs):
                output[infs] = np.interp(K[infs], K[~infs], output[~infs])
        except ValueError:
            output = np.clip(output, 0, 1e16)

        return np.clip(output, 0, 1e16)

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
                except TypeError:
                    continue
                err = np.sum((y - func(x, *res[0])) ** 2)
                if err < prev:
                    prev = err
                    best = b
            beta = best

        bounds = ((-1, 0), (1, np.inf))
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            res = curve_fit(func, x, y, p0, bounds=bounds)

        rho, volvol = res[0]
        alpha = cls.calibrate_alpha(beta, rho, volvol, ivol, S, K, t, rf, div)

        return alpha, beta, rho, volvol
