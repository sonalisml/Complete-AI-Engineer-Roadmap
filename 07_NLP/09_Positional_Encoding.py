import torch
import torch.nn as nn
import math


# ============================================================
# 1. POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len=100):

        super().__init__()

        # Create an empty positional encoding matrix
        # Shape: (max_len, embed_dim)
        pe = torch.zeros(max_len, embed_dim)

        # Position numbers: 0, 1, 2, ..., max_len-1
        # Shape: (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)

        # Frequency term
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) *
            (-math.log(10000.0) / embed_dim)
        )

        # Apply sine to even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd dimensions
        pe[:, 1::2] = torch.cos(position * div_term)

        # Store positional encoding as a non-trainable buffer
        self.register_buffer("pe", pe)


    def forward(self, x):

        # Number of tokens in the input
        seq_len = x.size(1)

        # Add positional information
        x = x + self.pe[:seq_len]

        return x


# ============================================================
# 2. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_dim):

        super().__init__()

        # Multi-Head Self-Attention
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

        # ----------------------------------------------------
        # Multi-Head Self-Attention
        # Q = K = V = x
        # ----------------------------------------------------

        attention_output, _ = self.attention(
            x,
            x,
            x
        )

        # ----------------------------------------------------
        # Residual Connection + LayerNorm
        # ----------------------------------------------------

        x = self.norm1(
            x + attention_output
        )

        # ----------------------------------------------------
        # Feed Forward Network
        # ----------------------------------------------------

        ffn_output = self.ffn(x)

        # ----------------------------------------------------
        # Residual Connection + LayerNorm
        # ----------------------------------------------------

        x = self.norm2(
            x + ffn_output
        )

        return x


# ============================================================
# 3. TRANSFORMER ENCODER
# ============================================================

class TransformerEncoder(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers
    ):

        super().__init__()

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            embed_dim
        )

        # Stack multiple Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                ff_dim
            )
            for _ in range(num_layers)
        ])


    def forward(self, x):

        # ----------------------------------------------------
        # Step 1: Add positional information
        # ----------------------------------------------------

        x = self.positional_encoding(x)

        # ----------------------------------------------------
        # Step 2: Pass through Transformer Blocks
        # ----------------------------------------------------

        for layer in self.layers:

            x = layer(x)

        return x


# ============================================================
# 4. TEST THE TRANSFORMER ENCODER
# ============================================================

# Batch size = 1
# Number of tokens = 4
# Embedding dimension = 8

x = torch.randn(1, 4, 8)

print("Input shape:")
print(x.shape)


# Create Transformer Encoder
encoder = TransformerEncoder(
    embed_dim=8,
    num_heads=2,
    ff_dim=32,
    num_layers=3
)


# Pass input through encoder
output = encoder(x)


print("\nOutput shape:")
print(output.shape)