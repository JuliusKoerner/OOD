import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()
        # Convolutional layers
        self.convs = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.fcs = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fcs(x)

        return x


class Conv_Net_Dropout(nn.Module):
    def __init__(self, prob=0.25):
        super(Conv_Net_Dropout, self).__init__()
        # Convolutional layers
        self.prob = prob
        self.convs = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.prob),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.prob),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.fcs = nn.Sequential(
            nn.Dropout(self.prob),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(self.prob),
            nn.Linear(128, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fcs(x)

        return x
