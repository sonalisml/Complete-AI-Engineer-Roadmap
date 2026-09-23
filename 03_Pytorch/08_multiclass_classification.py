import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
#============
#CREATE DATA
#===========
x = torch.tensor([
    # Class 0
    [1.0, 1.0],
    [1.2, 1.1],
    [1.1, 1.3],
    [1.3, 1.2],

    # Class 1
    [5.0, 5.0],
    [5.2, 5.1],
    [5.1, 5.3],
    [5.3, 5.2],

    # Class 2
    [9.0, 1.0],
    [9.2, 1.1],
    [9.1, 1.3],
    [9.3, 1.2]
])
y = torch.tensor([
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2
])
print("x shape is ", x.shape)
print("y shape is ", y.shape)

#======================================
#step 2: CREATE DATASET
#======================================
dataset = TensorDataset(X,y)
# =========================================================
# STEP 3: CREATE DATALOADER
# =========================================================
dataloader = DataLoader(
    dataset,
    batch_size = 4,
    shuffle = True
)

# =========================================================
# STEP 4: CREATE NEURAL NETWORK
# =========================================================
class MultiClassclassifier(nn.Module):
    def__init__(self):
        super().__init__()

        # 2 input features → 10 neurons
        self.layer1 = nn.Linear(2, 10)

        # Activation function
        self.relu = nn.ReLU()

        # 10 neurons → 3 output classes
        self.output = nn.Linear(10, 3)

      def forward(self, x):

        x = self.layer1(x)

        x = self.relu(x)

        x = self.output(x)

        return x
model = MultiClassClassifier()
print("\nModel:")
print(model)

#=============
#Step-loss and optimizer
#============
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.sgd(
    model.parameters(),
    lr =0.01
)
# =========================================================
# STEP 7: TRAINING
# =========================================================
epochs = 500
for batch_X, batch_y in dataloader:
    logits = model(batch_X)
    loss = criterion(logits, batch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
     if (epoch + 1) % 50 == 0:

        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss.item():.4f}"
        )

model.eval()
with torch.no_grad():
    #get logits
    logits = model(X)
    probabilities = torch.softmax(
        logits,
        dim =1
    )
    predictions = torch.argmax(
        logits,
        dim =1
    )

# =========================================================
# STEP 9: DISPLAY RESULTS
# =========================================================

print("\nLogits:")
print(logits)

print("\nProbabilities:")
print(probabilities)

print("\nPredictions:")
print(predictions)

print("\nActual labels:")
print(y)

correct = (predictions==y).sum().item()
total = len(y)
accuracy = correct/total
print(correct)
print(total)
print(accuracy)