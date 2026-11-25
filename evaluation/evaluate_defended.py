import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from models.baseline_model import BaselineCNN

batch_size = 128


def mnist_loader():
    dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_model(path, device):
    model = BaselineCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def accuracy(model, loader, device, attack=None):
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        inputs = attack(model, images, labels, device) if attack else images

        with torch.no_grad():
            predictions = model(inputs).argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100 * correct / total


if __name__ == "__main__":
    epsilon = 0.3
    step_size = 0.01
    steps = 40

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = mnist_loader()
    model = load_model("adv_model.pth", device)

    clean_acc = accuracy(model, loader, device)
    fgsm_acc = accuracy(model, loader, device, lambda m, x, y, d: fgsm_attack(m, x, y, epsilon, d))
    pgd_acc = accuracy(model, loader, device, lambda m, x, y, d: pgd_attack(m, x, y, epsilon, step_size, steps, d))

    print("Adversarially Trained Model")
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"FGSM Accuracy: {fgsm_acc:.2f}%")
    print(f"PGD Accuracy: {pgd_acc:.2f}%")
    print(f"FGSM Robustness Gap: {clean_acc - fgsm_acc:.2f}%")
    print(f"PGD Robustness Gap: {clean_acc - pgd_acc:.2f}%")
