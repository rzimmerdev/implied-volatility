import numpy as np
from scipy.optimize import minimize, OptimizeWarning
from .sabr import SABR


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
    def funcs(cls):
        return {
            "alpha": cls.alpha,
            "rho": cls.rho,
            "volvol": cls.volvol
        }

    @classmethod
    def fit(cls, func, size, candidates, bounds=None):  # Candidates \mathcal{S} = {(t, param^{t})}
        """
        Fit a parameter function to a set of possible candidates using the L-BFGS-B optimization method.
        A candidate is a tuple of (t, param^{t}),
        where param is one of the parameters of the SABR models (alpha, rho, volvol).

        :param func: Callable function to fit, can be either ParametricSABR.alpha, ParametricSABR.rho or ParametricSABR.volvol
        :param size: Array size of the parameter in question, 5 for alpha, 4 for rho and volvol
        :param candidates: Array of candidates
        :param bounds: Bounds for the optimization, for the alpha parameter,
        the bounds are [(1e-16, None) for _ in range(5)], etc.
        :return: The resulting optimized parameter, in the form of an array (size,), either P, Q or R
        """
        def error(param):
            return np.sum((np.subtract(func(candidates[:, 0], param),
                                       candidates[:, 1])) ** 2)

        initial_guess = np.random.rand(size)
        res = minimize(error, initial_guess, method='L-BFGS-B', bounds=bounds)

        return res.x

    @classmethod
    def fit_p(cls, candidates):
        bounds = [(1e-16, None) for _ in range(5)]
        return cls.fit(cls.alpha, 5, candidates, bounds)

    @classmethod
    def fit_q(cls, candidates):
        bounds = [(-1 + 1e-9, 1 - 1e-9), (None, None), (None, None), (0, None)]
        return cls.fit(cls.rho, 4, candidates, bounds)

    @classmethod
    def fit_r(cls, candidates):
        bounds = [(None, None), (None, None), (None, None), (None, None)]
        return cls.fit(cls.volvol, 4, candidates, bounds)

    @classmethod
    def fit_params(cls, candidates: dict):
        return cls.fit_p(candidates["alpha"]), cls.fit_q(candidates["rho"]), cls.fit_r(candidates["volvol"])

    def smooth_surface(self, S, K, T, rf=0.0, div=0.0, beta=0.5):
        """
        Generate a smooth surface of implied volatilities using the Parametric SABR models.
        Attributes P, Q, R are used to generate the surface.

        :param S: Fixed Spot price
        :param K: Array of possible strike prices
        :param T: Array of possible maturities
        :param rf: Risk-free rate
        :param div: Dividend yield
        :param beta: Beta parameter, default is 0.5
        :return: Array of implied volatilities, for each combination of K and T
        """
        if self.p is None or self.q is None or self.r is None:
            raise ValueError("Parameters not set")

        iv = np.zeros((len(T), len(K)))

        alpha = self.alpha(T, self.p)
        rho = self.rho(T, self.q)
        volvol = self.volvol(T, self.r)

        for idx, i in enumerate(T):
            iv[idx] = SABR.ivol(alpha[idx], beta, rho[idx], volvol[idx], S, K, i, rf, div)

        return iv  # shape: (len(T), len(K))

    @staticmethod
    def candidates(ivol, S, K, T, rf, div, beta):
        """
        Obtain parameters alpha, beta, rho and volvol for each maturity in T.
        The parameters are obtained by fitting a SABR models to the observed IVOL data.
        """
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

    @classmethod
    def optim(cls, ivol, S, K, T, rf, div, beta=None):
        """
        Finds the best combination P, Q, R for a Parametric SABR models, given daily observed IVOL data.

        :param ivol: Array of observed IVOL data
        :param S: Spot price
        :param K: Array of strike prices
        :param T: Array of maturities
        :param rf: Risk-free rate
        :param div: Dividend yield
        :param beta: Beta parameter, if None, it will also be optimized
        :return:
        """
        prev = np.inf
        best = beta

        if beta is None:
            for b in np.linspace(0.1, 0.9, 9):
                beta = b

                candidates = cls.candidates(ivol, S, K, T, rf, div, beta)
                p, q, r = cls.fit_params(candidates)
                sabr = cls(p, q, r)

                pred_iv = sabr.ivol(S, K, T, rf, div, beta)
                real_iv = ivol

                mask = np.abs(pred_iv - real_iv) < 6 * real_iv.std()
                pred_iv = pred_iv[mask]
                real_iv = real_iv[mask]

                err = mae(real_iv, pred_iv)
                if err < prev:
                    prev = err
                    best = b

        candidates = cls.candidates(ivol, S, K, T, rf, div, best)
        p, q, r = cls.fit_params(candidates)

        return p, q, r, best


def mae(x, y):
    return np.mean(np.abs(x - y))
