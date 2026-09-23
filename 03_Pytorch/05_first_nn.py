import torch
import torch.nn as nn

model = nn.Linear(1, 1)

print(model)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

for epoch in range(1000):
    # 1. Forward pass
    predictions = model(X)

    # 2. Calculate loss
    loss = criterion(predictions, y)

    # 3. Clear old gradients
    optimizer.zero_grad()

    # 4. Calculate new gradients
    loss.backward()

    # 5. Update model parameters
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(epoch + 1, loss.item())

model.eval()

with torch.no_grad():
    test_input = torch.tensor([[5.0]])
    prediction = model(test_input)

print(prediction.item())