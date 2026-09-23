import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
#=============================
#Step-1
#=============================
X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])
y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0]
])

# =========================================================
# STEP 2: CREATE DATASET
# =========================================================
dataset = TensorDataset(X,y)

# =========================================================
# STEP 3: CREATE DATALOADER
# =========================================================
dataloader = DataLoader(
    dataset,
    batch_size = 2,
    shuffle = True
)
# =========================================================
# STEP 4 Create model
# =========================================================
model = nn.Linear(1,1)


# STEP 5 loss
# =========================================================
criterion = nn.MSELoss()

# =========================================================
# STEP 6: OPTIMIZER
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01
)
# =========================================================
# STEP 7: TRAINING
# =========================================================
epochs = 10
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1)% 100 ==0:
        print(
            f"Epoch{epoch+1},"
            f"loss:{loss.item},"
        )
# =========================================================
# STEP 8: CHECK LEARNED PARAMETERS
# =========================================================

print("\nLearned weight:")
print(model.weight.item())

print("\nLearned bias:")
print(model.bias.item())

# =========================================================
# STEP 9: TEST THE MODEL
# =========================================================
model.eval()
with torch.no_grad():
     test_input = torch.tensor([[5.0]])
     predictions = model(test_input)
print(test_input.item())
print(prediction.item())