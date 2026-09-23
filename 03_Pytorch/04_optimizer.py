import torch

w = torch.tensor(2.0, requires_grad=True)

optimizer = torch.optim.SGD([w], lr=0.1)

loss = (w - 5) ** 2

loss.backward()

optimizer.step()

print(w.item())