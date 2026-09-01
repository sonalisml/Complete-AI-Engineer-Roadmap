import torch
from attention.cbam import CBAM

x = torch.randn(1,256,80,80)

cbam = CBAM(256)

output = cbam(x)

print("input shape", x.shape)
print("output shape", output.shape)