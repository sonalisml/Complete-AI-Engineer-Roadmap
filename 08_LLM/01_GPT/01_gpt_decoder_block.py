#Obj of the code- Create a GPT decoder block
import torch
import torch.nn as nn

class GPTDecoderBlock(nn.Module):
     def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        #Masked Multi Head Attention
        self.attention  = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            batch_first = True
        )
        #Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        #Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
     def forward(self, x):
        # Sequence length
        seq_len = x.size(1)
        # Causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len),
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


#TEST
x = torch.rand(1,4,8)
model = GPTDecoderBlock(
    embed_dim = 8,
    num_heads = 2,
    ff_dim =32
)
output = model(x)
print("Input shape", x.shape)
print("Output shape")
print(output.shape)