#Now the first neural network layer
import torch.nn as nn

layer = nn.Linear(2,3)

output = layer(x)

print("\nOutput:")
print(output)

print("\nOutput shape:")
print(output.shape)