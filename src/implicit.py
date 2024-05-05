import time

import torch
import torch.nn as nn


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


def forward_iteration(f, x0, max_iter=50, tol=1e-2):
    f0 = f(x0)
    res = []
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if (res[-1] < tol):
            break
    return f0, res


# Compare forward with anderson for a simple CNN
class CNN(nn.Module):
    def __init__(self, n_channels, hidden_size, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, hidden_size, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(hidden_size, n_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, hidden_size)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        self.relu = nn.ReLU()

    def forward(self, z, x):
        y = self.norm1(self.relu(self.conv1(z)))
        return self.norm3(self.relu(z + self.norm2(x + self.conv2(y))))


def main():
    # compare forward iteration with anderson acceleration
    X = torch.randn(10, 64, 32, 32)
    f = CNN(64, 128)

    # initial guess
    x0 = torch.zeros_like(X)

    # forward iteration
    initial_time = time.time()
    result, res = forward_iteration(lambda Z: f(Z, X), x0)
    print('Forward iteration time: ', time.time() - initial_time)

    # anderson acceleration
    initial_time = time.time()
    result, res = anderson(lambda Z: f(Z, X), x0)
    print('Anderson acceleration time: ', time.time() - initial_time)


if __name__ == '__main__':
    main()
