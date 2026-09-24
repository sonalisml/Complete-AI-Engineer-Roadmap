import torch
import torch.nn as nn

x = torch.tensor([1.0, 2.0, 3.0])
attention_output = torch.tensor([0.5, 1.0, 2.0])

#residual output
residual_output = x + attention_output

print("Input")
print(x)

print("\nAttention output")
print(attention_output)

print("Residual output")
print(residual_output)

#LayerNormalization
layer_norm = nn.LayerNorm(3)
normalized_output = layer_norm(residual_output)

print("\nnormalized output")
print(normalized_output)