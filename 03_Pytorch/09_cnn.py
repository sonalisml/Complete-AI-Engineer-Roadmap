import torch
import torch.nn as nn


# =========================================================
# STEP 1: CREATE CNN MODEL
# =========================================================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        # -----------------------------------------
        # First convolution
        # -----------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu1 = nn.ReLU()

        # -----------------------------------------
        # First pooling
        # -----------------------------------------

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # -----------------------------------------
        # Second convolution
        # -----------------------------------------

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu2 = nn.ReLU()

        # -----------------------------------------
        # Second pooling
        # -----------------------------------------

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # -----------------------------------------
        # Fully connected layer
        # -----------------------------------------

        self.fc = nn.Linear(
            32 * 8 * 8,
            10
        )


    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(self, x):

        x = self.conv1(x)

        x = self.relu1(x)

        x = self.pool1(x)

        x = self.conv2(x)

        x = self.relu2(x)

        x = self.pool2(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Fully connected layer
        x = self.fc(x)

        return x


# =========================================================
# CREATE MODEL
# =========================================================

model = CNN()

print(model)


# =========================================================
# TEST WITH A FAKE IMAGE BATCH
# =========================================================

# 4 images
# 3 channels (RGB)
# 32 × 32 pixels

X = torch.randn(4, 3, 32, 32)

print("\nInput shape:")
print(X.shape)


# =========================================================
# FORWARD PASS
# =========================================================

output = model(X)

print("\nOutput shape:")
print(output.shape)