import numpy as np

from ..sabr import ParametricSABR as PSABR


class SSV:
    @staticmethod
    def mc_combinations(point, param_candidates, k=20, p=0.7):
        n = int(p * len(param_candidates))

        indices = np.arange(len(param_candidates))
        combinations = np.array([param_candidates[np.random.choice(indices, n, replace=False)] for _ in range(k)])

        # at each row, at random index [0, n), set value to point
        combinations[:, np.random.randint(0, n)] = point

        return combinations

    @classmethod
    def _ssv(cls, point, candidates, optim_func, func, k=20, p=0.7):
        """
        Calculate SS value for a given point compared to a set of candidates.

        :param point: Point to compare, tuple of (t, param^{t})
        :param candidates: Array of candidates
        :param optim_func: Callable function to optimize, can be either ParametricSABR.alpha, ParametricSABR.rho or ParametricSABR.volvol
        :param k: MC sample size
        :param p: Percentage of candidates to use for combination sets
        :return:
        """
        # apply optim_func to each combination
        combinations = cls.mc_combinations(point, candidates, k, p)

        # vector (k, param_size) (param_size is given by optim_func)
        param_star = optim_func(candidates)
        param_tilde = np.vectorize(optim_func)(combinations)

        error = np.sum(np.abs(func(point[0], param_tilde) - func(point[0], param_star))) / k

        return error

    @classmethod
    def ssv(cls, candidates, optim_func, func, k=20, p=0.7):
        """
        Get SS values for each point in the candidates array.

        :param candidates:
        :param k:
        :param p:
        :return:
        """
        param_star = optim_func(candidates)

        ss_values = np.zeros(len(candidates))

        for idx, point in enumerate(candidates):
            combinations = cls.mc_combinations(point, candidates, k, p)
            param_tilde = np.array([optim_func(combinations[i]) for i in range(k)])

            ss_values[idx] = np.sum(np.abs(func(point[0], param_tilde) - func(point[0], param_star))) / k

        ss_values = np.clip(ss_values, 1e-16, 1e16)

        return ss_values, param_star

    @classmethod
    def z(cls, values, axis=0):
        """Z-Normalize a n-dimensional array."""
        return (values - np.mean(values, axis=axis)) / np.std(values, axis=axis)

    @classmethod
    def z_alpha(cls, candidates, k=20, p=0.7):
        ss_values, param_star = cls.ssv(candidates, PSABR.fit_p, PSABR.alpha, k, p)

        return cls.z(ss_values), param_star

    @classmethod
    def z_rho(cls, candidates, k=20, p=0.7):
        ss_values, param_star = cls.ssv(candidates, PSABR.fit_q, PSABR.rho, k, p)

        return cls.z(ss_values), param_star

    @classmethod
    def z_volvol(cls, candidates, k=20, p=0.7):
        ss_values, param_star = cls.ssv(candidates, PSABR.fit_r, PSABR.volvol, k, p)

        return cls.z(ss_values), param_star

    @classmethod
    def z_funcs(cls, iter_params: dict = None):
        if iter_params is None:
            iter_params = {
                "alpha": (10, 0.7),
                "rho": (10, 0.7),
                "volvol": (5, 0.7)
            }
        return {
            "alpha": lambda candidates: cls.z_alpha(candidates, *iter_params["alpha"]),
            "rho": lambda candidates: cls.z_rho(candidates, *iter_params["rho"]),
            "volvol": lambda candidates: cls.z_volvol(candidates, *iter_params["volvol"])
        }
