import torch
import torch.nn.functional as F


def pgd_attack(model, images, labels, epsilon, step_size, steps, device, random_start=False):
    clean_images = images.clone().detach().to(device)
    labels = labels.to(device)

    if random_start:
        noise = torch.empty_like(clean_images).uniform_(-epsilon, epsilon)
        perturbed_images = torch.clamp(clean_images + noise, 0, 1).detach()
    else:
        perturbed_images = clean_images.clone().detach()

    for _ in range(steps):
        perturbed_images.requires_grad_(True)

        outputs = model(perturbed_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        perturbation = perturbed_images + step_size * perturbed_images.grad.sign()
        perturbation = torch.clamp(perturbation - clean_images, -epsilon, epsilon)
        perturbed_images = torch.clamp(clean_images + perturbation, 0, 1).detach()

    return perturbed_images
