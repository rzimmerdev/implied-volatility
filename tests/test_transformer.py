import unittest

import torch
from torch import nn, optim

from src.models.transformer import TransformerEncoder


class MyTestCase(unittest.TestCase):
    x = torch.normal(0, 1, (64, 11, 4))
    y = torch.normal(5, 2, (64, 11, 4)) * 0.1

    model = TransformerEncoder(4, 4, 4, 2, 0.5, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    output = model(x)
    s_loss = criterion(output, y)
    print(f"Initial Loss: {s_loss.item()}")

    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item()}")

    loss = criterion(output, y)
    print(f"Final Loss: {loss.item()}")
    print(f"Improvement (%): {((s_loss - loss) / loss * 100).item():.2f}")


if __name__ == '__main__':
    unittest.main()
