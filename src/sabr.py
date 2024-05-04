import math
import numpy as np


class SABR:
    def __init__(self, a, b, r):
        self.a = a
        self.b = b
        self.r = r

    def f(self, S, K, T):
        # Hagan et al. (2002) forward price
        return S - self.a / self.b * (S * K) ** ((1 - self.b) / 2) * math.log(S / K)

    def z(self, S, K, T):
        # Hagan et al. (2002) z
        return self.a / self.b * (S * K) ** ((1 - self.b) / 2) * math.log(S / K)

    def ivol(self, S, K, T):
        # Hagan et al. (2002) implied volatility
        z = self.z(S, K, T)
        X = math.log((math.sqrt(1 - 2 * self.r * z + z ** 2) + z - self.r) / (1 - self.r))
        return self.a / (S * K) ** ((1 - self.b) / 2) * (1 + (1 - self.b) ** 2 / 24 * z ** 2 + (1 - self.b) ** 4 / 1920 * z ** 4) * T

def main():
    # draw simple volatility smile
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")

    sabr = SABR(0.1, 0.2, 0.1)

    S = 100
    T = np.linspace(0.1, 2, 10)
    K = np.array([
        np.random.randint(80, 120, 100) for _ in T
    ])

    IV = np.array([
        [sabr.ivol(S, k, t) for k in K[i]] for i, t in enumerate(T)
    ])

    # flatten K and IV, and make T the same size
    T = np.array([
        [t] * len(K[i]) for i, t in enumerate(T)
    ]).flatten()

    K = K.flatten()
    IV = IV.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(K, T, IV, cmap='viridis')

    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('IV')

    plt.show()


if __name__ == '__main__':
    main()
