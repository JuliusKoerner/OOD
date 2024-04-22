import torch
import torch.nn as nn
import torch.nn.functional as F


def load_model(name):
    models = {
        "Conv_Net": Conv_Net,
        "Conv_Net_Dropout": Conv_Net_Dropout,
        "Conv_Net_FC_Dropout": Conv_Net_FC_Dropout,
    }
    return models[name]


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
    """
    Convolutional Neural Network with dropout layers before every layer."""

    def __init__(self, prob=0.5):
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

    def set_dropout(self, prob):
        self.prob = prob
        for layer in self.convs:
            if isinstance(layer, nn.Dropout):
                layer.p = self.prob
        for layer in self.fcs:
            if isinstance(layer, nn.Dropout):
                layer.p = self.prob

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fcs(x)

        return x


class Conv_Net_FC_Dropout(Conv_Net_Dropout):
    """
    Convolutional Neural Network with dropout layers only before the fully connected layers.
    """

    def __init__(self, prob=0.5):
        super(Conv_Net_FC_Dropout, self).__init__()
        # Convolutional layers
        self.prob = prob
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
