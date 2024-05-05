import torch
import torch.nn as nn
from torch import optim


class SelfAttention(nn.Module):
    """
    Attention for the transformer model
    Accepts continuous data

    """
    def __init__(self, in_features, heads):
        super(SelfAttention, self).__init__()
        self.head_dim = in_features // heads
        self.in_features = in_features
        self.heads = heads

        assert (self.head_dim * heads == in_features)

        self.W_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_q = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.seq = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, value, key, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        value = value.reshape(N, value_len, self.heads, self.head_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.W_v(value)
        keys = self.W_k(key)
        queries = self.W_q(query)

        # (N, value_len, heads, head_dim) * (N, key_len, heads, head_dim) -> (N, heads, value_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)
        # (N, heads, query_len, key_len) * (N, value_len, heads, head_dim) -> (N, query_len, heads, head_dim)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.in_features)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, in_features, heads, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(in_features, heads)
        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(in_features)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features, forward_expansion * in_features),
            nn.ReLU(),
            nn.Linear(forward_expansion * in_features, in_features)
        )

    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask)

        x = self.norm1(attention + value)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


class Transformer(nn.Module):
    def __init__(self, in_features, heads, num_layers, out_features, forward_expansion=None, dropout=None):
        super(Transformer, self).__init__()
        if forward_expansion is None:
            forward_expansion = 4
        if dropout is None:
            dropout = 1e-6

        self.layers = nn.ModuleList([
            TransformerBlock(in_features, heads, forward_expansion) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        for layer in self.layers:
            value = layer(value, key, query, mask)
        out = self.fc(value)
        return out


def main():
    x = torch.rand(64, 10, 256)
    y = torch.rand(64, 10, 256)

    model = Transformer(256, 8, 4, 256)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    for epoch in range(10):
        optimizer.zero_grad()
        output = model(x, x, x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} loss: {loss.item()}")


if __name__ == "__main__":
    main()
