import numpy as np
from scipy.optimize import minimize, OptimizeWarning
from tqdm import tqdm
from sabr import SABR


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

    def save(self, beta=None):
        # save p, q, r and beta to a csv
        np.savetxt('sabr.csv', np.concatenate([self.p, self.q, self.r, [beta]]), delimiter=',')

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
        return np.clip(q[0] + q[1] * t + q[2] * np.exp(-q[3] * t), -1 + 1e-9, 1 - 1e-9)

    @staticmethod
    def volvol(t, r):  # r = (4,)
        return np.clip(
            r[0] + r[1] * np.power(np.maximum(t, 1e-16), r[2] + 1e-16) * np.exp(np.clip(r[3] * t + 1e-16, -1e2, 5e2)),
            1e-16, None)

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

        for idx, i in enumerate(T):
            iv[idx] = SABR.ivol(alpha[idx], beta, rho[idx], volvol[idx], S, K, i, rf, div)

        return iv  # shape: (len(T), len(K))

    def fit_candidates(self, ivol, S, K, T, rf, div, beta, logger=None):
        maturities = np.unique(T)

        candidates = {"alpha": [], "rho": [], "volvol": []}

        for idx, t in enumerate(maturities):
            mask = T == t
            masked_K = K[mask]
            masked_ivol = ivol[mask]

            if len(masked_ivol) < 5:
                continue

            try:
                alpha, _, rho, volvol = SABR.fit_sabr(masked_ivol, S, masked_K, t, rf, div, beta)
            except OptimizeWarning:
                # logger.warning(f"Optimization failed at {t}")
                continue
            except RuntimeError:
                # logger.info(f"Parameter estimation failed at {t}")
                continue

            candidates["alpha"].append((t, alpha))
            candidates["rho"].append((t, rho))
            candidates["volvol"].append((t, volvol))

        return {candidate: np.array(candidates[candidate]) for candidate in candidates}

    def optim_beta(self, ivol, S, K, T, rf, div, beta, logger=None):
        prev = np.inf
        best = None

        for b in tqdm(np.linspace(0.1, 0.9, 9) if beta is None else [beta]):
            beta = b

            candidates = ParametricSABR.fit_candidates(ivol, S, K, T, rf, div, beta, logger)
            p, q, r = ParametricSABR.fit_params(candidates)
            sabr = ParametricSABR(p, q, r)

            pred_iv = sabr.ivol(S, K, T, rf, div, beta)
            real_iv = ivol

            mask = np.abs(pred_iv - real_iv) < 6 * real_iv.std()
            pred_iv = pred_iv[mask]
            real_iv = real_iv[mask]

            err = mae(real_iv, pred_iv)
            if err < prev:
                prev = err
                best = b

        return best


def mae(x, y):
    return np.mean(np.abs(x - y))
