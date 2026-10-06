import torch
import torch.nn as nn


# ============================================================
# GPT Decoder Block
# ============================================================

class GPTDecoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self, x):
        seq_len = x.size(1)
        mask =torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device = x.device
            ),
             diagonal =1
        ).bool()
        attention_output,_= self.attention(
            x,
            x,
            x,
            attn_mask=mask
        )
        x = self.norm1(x + attention_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


# ============================================================
# GPT Model
# ============================================================

class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            embed_dim
        )

        self.layers = nn.ModuleList([
            GPTDecoderBlock(
                embed_dim,
                num_heads,
                ff_dim
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.shape

        token_embeddings = self.token_embedding(input_ids)

        positions = torch.arange(
            seq_len,
            device=input_ids.device
        )

        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        return x


# ============================================================
# LM HEAD
# ============================================================

class LMHead(nn.Module):

    def __init__(self, embed_dim, vocab_size):
        super().__init__()

        self.linear = nn.Linear(
            embed_dim,
            vocab_size
        )

    def forward(self, x):

        logits = self.linear(x)

        return logits


# ============================================================
# Test
# ============================================================

vocab_size = 10000
embed_dim = 64

gpt = GPTModel(
    vocab_size=10000,
    max_seq_len=128,
    embed_dim=64,
    num_heads=4,
    ff_dim=256,
    num_layers=4
)

lm_head = LMHead(
    embed_dim=embed_dim,
    vocab_size=vocab_size
)

input_ids = torch.tensor([
    [10, 25, 78, 91, 42]
])

hidden_states = gpt(input_ids)

logits = lm_head(hidden_states)

print("Input shape       :", input_ids.shape)
print("Hidden state shape:", hidden_states.shape)
print("Logits shape      :", logits.shape);