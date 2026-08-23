"""
Step 1: Train a simple image classifier (CNN) on MNIST.
This is the "victim" model that we will attack with FGSM/PGD in later steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3
MODEL_PATH = "mnist_cnn.pt"


# A small CNN: 2 conv layers + 2 fully-connected layers.
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)                     # raw logits (no softmax)


def get_dataloaders():
    transform = transforms.ToTensor()  # pixels scaled to [0, 1]
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def train(model, train_loader, optimizer):
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{EPOCHS} - train loss: {avg_loss:.4f}")


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy


def main():
    print(f"Using device: {DEVICE}")
    train_loader, test_loader = get_dataloaders()

    model = SimpleCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train(model, train_loader, optimizer)

    clean_accuracy = evaluate(model, test_loader)
    print(f"\nClean test accuracy: {clean_accuracy * 100:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
