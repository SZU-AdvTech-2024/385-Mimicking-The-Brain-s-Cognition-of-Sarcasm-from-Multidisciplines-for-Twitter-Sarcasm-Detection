import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiheadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N = x.shape[0]  # batch size
        length = x.shape[1]  # sequence length

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        # Split into heads
        values = values.view(N, length, self.heads, self.head_dim).transpose(1, 2)
        keys = keys.view(N, length, self.heads, self.head_dim).transpose(1, 2)
        queries = queries.view(N, length, self.heads, self.head_dim).transpose \
            (1, 2)

        # Calculate attention
        energy = torch.einsum("nhqd,nhkd->nhqk", [queries, keys])
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Get the weighted sum of values
        out = torch.einsum("nhql,nhld->nhqd", [attention, values]).reshape(N, length, self.embed_size)

        return self.fc_out(out)


class MultiheadCrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiheadCrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        N = query.shape[0]  # batch size
        length_q = query.shape[1]  # sequence length of query
        length_k = key.shape[1]  # sequence length of key/value

        values = self.values(value)
        keys = self.keys(key)
        queries = self.queries(query)

        # Split into heads
        values = values.view(N, length_k, self.heads, self.head_dim).transpose(
            1, 2)
        keys = keys.view(N, length_k, self.heads, self.head_dim).transpose(1, 2)
        queries = queries.view(N, length_q, self.heads,
                               self.head_dim).transpose(1, 2)

        # Calculate attention
        energy = torch.einsum("nhqd,nhkd->nhqk", [queries, keys])
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Get the weighted sum of values
        out = torch.einsum("nhql,nhld->nhqd", [attention, values]).reshape(N,
                                                                           length_q,
                                                                           self.embed_size)

        return self.fc_out(out)



if __name__ == "__main__":

    # Example usage
    embed_size = 256
    heads = 8
    """
    self_attention = MultiheadSelfAttention(embed_size, heads)
    x = torch.rand(10, 1, embed_size)  # Batch size of 10, sequence length of 20
    output = self_attention(x)
    """
    # Example usage
    cross_attention = MultiheadCrossAttention(embed_size, heads)
    query = torch.rand(10, 15, embed_size)  # Query with sequence length of 15
    key = torch.rand(10, 20, embed_size)  # Key/Value with sequence length of 20
    value = key
    output = cross_attention(query, key, value)
    print(output.shape)
