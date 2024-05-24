import numpy as np

from src.sabr import ParametricSABR
from src.transformer import Transformer

from scipy.optimize import minimize


class SSV:  # SS value model
    def __init__(self, sabr: ParametricSABR):
        self.sabr = sabr

    def get_combinations(self, point, candidates, k=1000, p=0.4):
        # param is alpha^{t} := (alpha, t)
        # candidates is a np.array of shape (n, 2), dim 0 = alpha, rho or volvol, dim 1 = tenor
        # generate k random combinations of size int(p * len(candidates)) that contain param[0]
        combination_size = int(p * len(candidates))

        # Generate k random combinations of size combination_size that contain point
        # Use numpy to get combinations without repetition in the same combination
        combinations = np.zeros((k, combination_size, 2))

        for i in range(k):
            combinations[i] = np.random.choice(candidates[:, 0], size=combination_size, replace=False)
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

    def __getitem__(self, item):





class SST(SSV):  # SSV + Transformer
    pass
