import numpy as np

from src.sabr import ParametricSABR
from src.transformer import Transformer

from scipy.optimize import minimize


class SSV:  # SS value model
    def __init__(self, sabr: ParametricSABR):
        self.sabr = sabr

    def mc_combinations(self, alpha, candidates, n=1000, combination_size=5):
        p_star = self.sabr.p_star(candidates)

        # generate random combination sets such that alpha ∈ combination ∈ candidates
        if alpha not in candidates:
            raise ValueError(f"alpha {alpha} not in candidates {candidates}")

        candidates_without_alpha = candidates[candidates != alpha]
        combinations = np.random.choice(candidates_without_alpha, size=(n, combination_size - 1))
        combinations = np.hstack((combinations, np.repeat(alpha, n).reshape(-1, 1)))

        return combinations


class SST(SSV):  # SSV + Transformer
    pass

