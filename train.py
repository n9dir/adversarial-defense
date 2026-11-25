import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.baseline_model import BaselineCNN

batch_size = 128
epochs = 5
learning_rate = 1e-3


def mnist_loader():
    dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loader = mnist_loader()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} - loss: {running_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "baseline_model.pth")
    print("Saved baseline_model.pth")


if __name__ == "__main__":
    train()
