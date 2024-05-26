import numpy as np

from src.sabr import ParametricSABR

from tqdm import tqdm


class SSV:  # SS value model
    def __init__(self, rf = None, dv = None, sabr: ParametricSABR = None):
        self.sabr = sabr if sabr is not None else ParametricSABR(rf, dv)

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


class SST(SSV):  # SSV + Transformer
    def __init__(self, rf = None, dv = None, sabr: ParametricSABR = None):
        super().__init__(rf, dv, sabr)


    def get_inputs(self, func, corrected, candidates, k=1000, p=0.4):
        # 4 dimensions:
        # 0: z-values
        # 1: point value (alpha, rho, or volvol)
        # 2: func(t, p*_prev) for point tenor
        # 3: 1 if is raw, 0 if adjusted
        inputs = np.zeros((len(candidates), 4))

        z_values = self.z_alpha(candidates, k, p)

        inputs[:, 0] = z_values
        inputs[:, 1] = candidates
        inputs[:, 2] = func(candidates[:, 1], corrected)




def main():
    n = 20
    observations = np.random.rand(n)
    tenors = np.linspace(0.1, 10, n)

    candidates = np.array([[observations[i], tenors[i]] for i in range(n)])

    risk_free = 0.05
    dividend = 0.02

    ssv = SSV(risk_free, dividend)

    z_values = ssv.z_alpha(candidates, k=10)

    print(z_values)


if __name__ == "__main__":
    main()
