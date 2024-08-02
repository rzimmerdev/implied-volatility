import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Attention for the transformer models.
    Accepts continuous data.
    """

    def __init__(self, in_features, heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_dim = in_features // heads
        self.in_features = in_features
        self.heads = heads

        assert self.head_dim * heads == in_features, "in_features must be divisible by heads"

        self.W_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_q = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(in_features, in_features)

    def forward(self, value, key, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        value = value.reshape(N, value_len, self.heads, self.head_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.W_v(value)
        keys = self.W_k(key)
        queries = self.W_q(query)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.in_features)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, in_features, heads, forward_expansion, num_layers=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = SelfAttention(in_features, heads)
        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(in_features)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features, forward_expansion * in_features),
            nn.ReLU(),
            nn.Linear(forward_expansion * in_features, in_features)
        )

    def forward(self, x, mask=None):
        # x is batch_size x seq_len x in_features
        attention = self.attention(x, x, x, mask)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self,
                 in_features,
                 heads,
                 num_blocks,
                 num_layers,
                 forward_expansion,
                 out_features=None,
                 dropout=0.5,
                 *args, **kwargs):
        """
        Transformer Encoder for continuous data.

        Args:
            in_features: int, number of input features
            heads: int, number of heads in the self-attention layer
            num_blocks: int, number of transformer blocks
            num_layers: int, number of layers in the feed forward network
            forward_expansion: int, expansion factor for the feed forward network
        """
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList([
            TransformerBlock(in_features, heads, forward_expansion, num_layers) for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        if out_features is not None:
            self.fc_out = nn.Linear(in_features, forward_expansion)
        else:
            self.fc_out = None

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        if self.fc_out is not None:
            x = self.fc_out(x)
        return x
