import torch
import torch.nn as nn
from torch import optim


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )  # (N, query_len, embed_size)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            num_layers,
            forward_expansion,
            dropout,
            max_length,
            vocab_size,
            device,
    ):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.token_embeddings(x) + self.position_embeddings(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = self.fc_out(out)
        return out


def main():
    vocab_size = 10
    max_length = 5
    embed_size = 64
    heads = 2
    num_layers = 1
    forward_expansion = 4
    dropout = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a sample input sequence
    input_sequence = torch.randint(0, vocab_size, (1, max_length)).to(device)
    mask = torch.ones(1, max_length).to(device)  # Mask for padding tokens

    # Create the Transformer model
    model = Transformer(
        embed_size,
        heads,
        num_layers,
        forward_expansion,
        dropout,
        max_length,
        vocab_size,
        device,
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    def train_model(model, input_sequence, mask, optimizer, criterion, num_epochs=10):
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = model(input_sequence, mask)
            loss = criterion(output.reshape(-1, vocab_size), input_sequence.reshape(-1))
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Test the model
    train_model(model, input_sequence, mask, optimizer, criterion)


if __name__ == "__main__":
    main()
