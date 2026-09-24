import torch
import torch.nn as nn

x= torch.randn(1,4,8)

mha = nn.MultiheadAttention(
    embed_dim = 8,
    num_heads = 2,
    batch_first = True
)

output, attention_weights = mha(x,x,x)

print("Input shape:")
print(x.shape)
print("\nOutput shape:")
print(output.shape)
print("attention weight shape")
print(attention_weights.shape)