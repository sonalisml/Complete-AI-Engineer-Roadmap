import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# =========================================================
# STEP 1: DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================================================
# STEP 2: TRANSFORMS
# =========================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# =========================================================
# STEP 3: DOWNLOAD TRAINING DATASET
# =========================================================

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# =========================================================
# STEP 4: DOWNLOAD TEST DATASET
# =========================================================

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# =========================================================
# STEP 5: CREATE DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# =========================================================
# STEP 6: DEFINE CNN
# =========================================================

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc = nn.Linear(
            32 * 8 * 8,
            10
        )


    def forward(self, x):

        x = self.conv1(x)

        x = self.relu1(x)

        x = self.pool1(x)

        x = self.conv2(x)

        x = self.relu2(x)

        x = self.pool2(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.fc(x)

        return x


# =========================================================
# STEP 7: CREATE MODEL
# =========================================================

model = CNN()

model = model.to(device)

print("\nModel:")
print(model)


# =========================================================
# STEP 8: LOSS FUNCTION
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# STEP 9: OPTIMIZER
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================================================
# STEP 10: TRAINING
# =========================================================

epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # -----------------------------------------
        # Forward pass
        # -----------------------------------------

        outputs = model(images)

        # -----------------------------------------
        # Calculate loss
        # -----------------------------------------

        loss = criterion(
            outputs,
            labels
        )

        # -----------------------------------------
        # Clear gradients
        # -----------------------------------------

        optimizer.zero_grad()

        # -----------------------------------------
        # Backpropagation
        # -----------------------------------------

        loss.backward()

        # -----------------------------------------
        # Update weights
        # -----------------------------------------

        optimizer.step()

        running_loss += loss.item()


    average_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.4f}"
    )


# =========================================================
# STEP 11: EVALUATION
# =========================================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


accuracy = 100 * correct / total


print("\nTest Accuracy:")
print(f"{accuracy:.2f}%")