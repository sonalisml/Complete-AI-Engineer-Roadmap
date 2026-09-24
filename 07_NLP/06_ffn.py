import torch
import torch.nn as nn

#One sentence, 4 tokens, each token has 8 features
x = torch.randn(1,4,8)

#FeedForward network
ffn = nn.Sequential(
    nn.Linear(8,32),
    nn.ReLU(),
    nn.Linear(32,8)
)

output = ffn(x)

print("Input shape")
print(x)

print("output shape")
print(output)
