import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()
        # Convolutional layers
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.fcs = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fcs(x)
        return x
