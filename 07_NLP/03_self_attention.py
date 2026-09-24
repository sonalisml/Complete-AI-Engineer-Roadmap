import torch
import torch.nn.functional as F

X = torch.tensor([
    [1.0, 0.0],   # robot
    [0.0, 1.0],   # moves
    [1.0, 1.0]    # forward
])

# Simplified example
Q = X
K = X
V = X

# 1. Calculate attention scores
scores = Q @ K.T

print("Attention scores:")
print(scores)

# 2. Scale
d_k = K.shape[1]

scaled_scores = scores / torch.sqrt(
    torch.tensor(d_k, dtype=torch.float32)
)

print("\nScaled scores:")
print(scaled_scores)


# 3. Softmax
attention_weights = F.softmax(
    scaled_scores,
    dim=-1
)

print("\nAttention weights:")
print(attention_weights)


# 4. Weighted sum of values
output = attention_weights @ V

print("\nAttention output:")
print(output)