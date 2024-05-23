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

    def p_coefficients(self, S_alpha, alpha_s):
        def objective(p):
            errors = self.sabr.alpha(S_alpha, p) - alpha_s
            return np.sum(errors**2)
        
        initial_guess = np.zeros(4)
        result = minimize(objective, initial_guess)
        return result.x


    def get_ssvalue(self, alpha_t, S_alpha, alpha_s, candidates):
        combinations = self.mc_combinations(alpha_t, candidates)
        losses = []
        p_star = self.sabr.p_star(candidates)

        for subset in combinations:
            p_hat = self.p_coefficients(subset, alpha_s)
            loss = np.sum(np.abs(self.sabr.alpha(S_alpha, p_hat) - self.sabr.alpha(S_alpha, p_star)))
            losses.append(-loss)
        
        return np.mean(losses)


class SST(SSV):  # SSV + Transformer
    pass
