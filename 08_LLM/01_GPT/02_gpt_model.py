import torch
import torch.nn as nn


# ============================================================
# GPT Decoder Block
# ============================================================

class GPTDecoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()

        # Masked Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        # Sequence length
        seq_len = x.size(1)

        # Causal mask
        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            ),
            diagonal=1
        ).bool()

        # Masked self-attention
        attention_output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=mask
        )

        # Residual + LayerNorm
        x = self.norm1(x + attention_output)

        # Feed Forward Network
        ffn_output = self.ffn(x)

        # Residual + LayerNorm
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

        # Token Embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        # Positional Embedding
        self.position_embedding = nn.Embedding(
            max_seq_len,
            embed_dim
        )

        # Stack multiple GPT decoder blocks
        self.layers = nn.ModuleList([
            GPTDecoderBlock(
                embed_dim,
                num_heads,
                ff_dim
            )
            for _ in range(num_layers)
        ])

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeddings = self.token_embedding(input_ids)

        # Position IDs
        positions = torch.arange(
            seq_len,
            device=input_ids.device
        )

        # Position embeddings
        position_embeddings = self.position_embedding(positions)

        # Combine token + position information
        x = token_embeddings + position_embeddings

        # Pass through GPT decoder blocks
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.final_norm(x)

        return x


# ============================================================
# Test the GPT Model
# ============================================================

vocab_size = 10000
max_seq_len = 128
embed_dim = 64
num_heads = 4
ff_dim = 256
num_layers = 4

model = GPTModel(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_layers=num_layers
)


# Example token IDs
input_ids = torch.tensor([
    [10, 25, 78, 91, 42]
])


output = model(input_ids)


print("Input shape :", input_ids.shape)
print("Output shape:", output.shape)