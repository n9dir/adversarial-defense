import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self._feature_size(), 128)
        self.fc2 = nn.Linear(128, 10)

    def _features(self, x):
        x = F.relu(self.conv1(x))
        return self.pool(F.relu(self.conv2(x)))

    def _feature_size(self):
        with torch.no_grad():
            return self._features(torch.zeros(1, 1, 28, 28)).numel()

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
