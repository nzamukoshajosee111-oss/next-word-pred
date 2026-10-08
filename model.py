import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # True means a position is blocked from seeing that future position
        mask = torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool),
            diagonal=1
        )

        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        scores = scores.masked_fill(
            self.causal_mask[:seq_len, :seq_len],
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)
        attention = weights @ v

        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, seq_len, embed_dim)

        return self.proj(attention)


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, block_size,
                 use_residual=True):
        super().__init__()

        self.use_residual = use_residual

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = CausalSelfAttention(
            embed_dim, num_heads, block_size
        )

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        # Attention sublayer
        attention_output = self.attention(self.ln1(x))

        if self.use_residual:
            x = x + attention_output
        else:
            x = attention_output

        # Feed-forward sublayer
        ffn_output = self.ffn(self.ln2(x))

        if self.use_residual:
            x = x + ffn_output
        else:
            x = ffn_output

        return x


class TinyWordGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size=12,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        use_residual=True
    ):
        super().__init__()

        self.block_size = block_size
        self.use_residual = use_residual

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                block_size,
                use_residual
            )
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape

        positions = torch.arange(seq_len, device=idx.device)

        x = (
            self.token_embedding(idx)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        logits = self.lm_head(x)                # (batch, seq, vocab)

        loss = None

        if targets is not None:
            # ---- FIX: predict only the NEXT token ----
            # preprocess.py gives us one target per example,
            # so we only use the logits at the last position.
            last_logits = logits[:, -1, :]      # (batch, vocab)
            loss = F.cross_entropy(last_logits, targets)
            # -------------------------------------------

        return logits, loss


print("Transformer model defined successfully!")