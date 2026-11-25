import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from attacks.pgd import pgd_attack
from models.baseline_model import BaselineCNN

batch_size = 128
epochs = 6
learning_rate = 1e-3
epsilon = 0.3
step_size = 0.03
steps = 10


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

    if os.path.exists("baseline_model.pth"):
        model.load_state_dict(torch.load("baseline_model.pth", map_location=device))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loader = mnist_loader()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            adversarial_images = pgd_attack(model, images, labels, epsilon, step_size, steps, device, random_start=True)

            optimizer.zero_grad()
            clean_loss = loss_fn(model(images), labels)
            adversarial_loss = loss_fn(model(adversarial_images), labels)
            loss = 0.25 * clean_loss + 0.75 * adversarial_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch % 50 == 0:
                print(f"Epoch {epoch}/{epochs} - batch {batch}/{len(loader)}")

        print(f"Epoch {epoch}/{epochs} - loss: {running_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "adv_model.pth")
    print("Saved adv_model.pth")


if __name__ == "__main__":
    train()
