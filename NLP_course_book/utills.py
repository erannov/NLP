import json
import torch
from torch import nn
import stanza

nlp = stanza.Pipeline("en")


# GELU activation function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = json.load(open("output/text_stanza.json", "r", encoding="utf-8"))

    vocab = sorted(set(tokens))
    print(f"Vocabulary size: {len(vocab)}")
    word2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2word = {idx: token for token, idx in enumerate(vocab)}
    return text, tokens, vocab, word2idx, idx2word


class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # New
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )  # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # New batch dimension b
        # For inputs where `num_tokens` exceeds `context_length`, this will result in errors
        # in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs
        # do not exceed `context_length` before reaching this forward method.
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)  # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # New

        context_vec = attn_weights @ values
        return context_vec


# stand alone multi-head attention, with weight split
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
        hidden_dim=None,
    ):
        super(MultiHeadAttention, self).__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.d_head = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.context_length = context_length
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Linear projections
        queries = (
            self.W_query(x)
            .view(b, num_tokens, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        keys = (
            self.W_key(x)
            .view(b, num_tokens, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        values = (
            self.W_value(x)
            .view(b, num_tokens, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

        # Compute attention scores
        scores = queries @ keys.transpose(-2, -1)
        scores.masked_fill_(self.mask[:num_tokens, :num_tokens], float("-inf"))
        weights = nn.functional.softmax(scores / self.d_head**0.5, dim=-1)
        weights = self.dropout(weights)

        # Compute context vector as weighted sum of values
        context_vec = weights @ values
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, -1)
        return context_vec
