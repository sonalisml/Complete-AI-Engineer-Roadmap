import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()

        #MultiHead attention
        self.attention = nn.MultiheadAttention(
             embed_dim=embed_dim,
             num_heads=num_heads,
             batch_first = True
           )
        #FFN-Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        #Layernorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self,x):
        #----Multihead attention
        attention_output,_ = self.attention(x,x,x)

        #---residual+layernorm
        x = self.norm1(x+attention_output)
        
        #---ffn
        ffn_output = self.ffn(x)
        
        #---residual+layernorm
        x= self.norm2(x+ffn_output)

        return x

#=======================================
# Test the Transformer Block
#=======================================
x = torch.randn(1,4,8)
model = TransformerBlock(
    embed_dim=8,
    num_heads=2,
    ff_dim=32
)

output = model(x) 

print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)