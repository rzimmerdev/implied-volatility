import torch
import torch.nn as nn
import torch.nn.functional as F


# Anderson Optimizator for fixed point process, for implicit neural networks
def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    k = 1
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]

        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]  # (bsz x n))

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:, k % m].view_as(x0), res


class ImplicitLinear(nn.Module):
    def __init__(self, in_features, out_features, f: callable, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.linear = nn.Linear(in_features, out_features)
        self.f = f

    def forward(self, x):
        with torch.no_grad():
            # use anderson + self.f + self.linear to solve the implicit function
            def f_(x):
                return self.linear(x)

            x, _ = anderson(f_, x, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0)

        return F.linear(x, self.weight, self.bias)


def main():
    l = ImplicitLinear(10, 20, torch.relu)

    x = torch.randn(10, 10)
    print(l(x).shape)

    y = torch.randn(10, 20)

    loss = F.mse_loss(l(x), y)

    print(loss)


if __name__ == "__main__":
    main()
