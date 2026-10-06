import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
     def __init__(self, embed_dim, num_heads, ff_dim):
          super().__init__()

          self.attention = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            batch_first = True
          )
          self.ffn = nn.Sequential(
            nn.Linear(embed_dim,ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim,embed_dim)
          )
          self.norm1= nn.LayerNorm(embed_dim)
          self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self,x):
        #multihead attention
        attention_output,_ = self.attention(x,x,x)
        #residual connection+layernorm
        x = self.norm1(x + attention_output)
        #ffn
        ffn_output = self.ffn(x)
        #residual+layer_norm
        x = self.norm2(x+ffn_output)
        return x


class TransformerEncoder(nn.module):
      def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
          super().__init__()

          self.layers = nn.ModuleList([
               TransformerBlock(embed_dim, num_heads, ff_dim)
               for _ in range(num_layers)
          ])

    def forward(self, x):

         for layer in self.layers:
             x = layer(x)

         return x