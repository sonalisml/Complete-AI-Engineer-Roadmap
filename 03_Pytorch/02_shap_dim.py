import torch

x = torch.tensor([[10, 20], [30, 40]])

print(x.shape)
print(x.ndim)
print(x[1, 0])



y = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print(y[0])       # First row
print(y[1, 2])    # Second row, third column
print(y[:, 1])    # All rows, second column
print(y[0, :2])   # First row, first two columns