import numpy as np
from tqdm import tqdm

from src.sabr.sabr import ParametricSABR


class SSV:  # SS value model
    def __init__(self, rf=None, dv=None):
        self.sabr = ParametricSABR(rf, dv)

    def get_combinations(self, point, candidates, k=1000, p=0.4):
        # param is alpha^{t} := (alpha, t)
        # candidates is a np.array of shape (n, 2), dim 0 = alpha, rho or volvol, dim 1 = tenor
        # generate k random combinations of size int(p * len(candidates)) that contain param[0]
        combination_size = int(p * len(candidates))

        # Generate k random combinations of size combination_size that contain point
        # Use numpy to get combinations without repetition in the same combination
        combinations = np.zeros((k, combination_size, 2))

        for i in range(k):
            indices = np.arange(len(candidates))
            combinations[i] = candidates[np.random.choice(indices, combination_size, replace=False)]
            combinations[i][np.random.randint(0, combination_size)] = point

        return combinations

    @property
    def param_tilde(self):
        return self.sabr.param_star

    def ss_value(self, point, candidates, func, size, k=1000, p=0.4):
        sumation = 0

        combinations = self.get_combinations(point, candidates, k, p)
        p_star = self.sabr.param_star(func, size, candidates)

        for combination in combinations:
            p_tilde = self.param_tilde(func, size, combination)

            sumation -= np.abs(
                func(point[1], p_tilde) - func(point[1], p_star)
            )

        return sumation / k

    def z_values(self, func, size, candidates, k=1000, p=0.4):

        ss_values = np.zeros(len(candidates))

        for idx, point in tqdm(enumerate(candidates), total=len(candidates)):
            ss_values[idx] = self.ss_value(point, candidates, func, size, k, p)

        mean = np.mean(ss_values)
        std = np.std(ss_values)

        z_values = (ss_values - mean) / std

        return z_values

    def z_alpha(self, candidates, k=1000, p=0.4):
        return self.z_values(self.sabr.alpha, 5, candidates, k, p)

    def z_rho(self, candidates, k=1000, p=0.4):
        return self.z_values(self.sabr.rho, 4, candidates, k, p)

    def z_volvol(self, candidates, k=1000, p=0.4):
        return self.z_values(self.sabr.volvol, 4, candidates, k, p)
