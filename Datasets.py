import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


def load(name):
    datasets = {"MNIST": MNISTDataset}
    return datasets[name]


class MNISTDataset(Dataset):
    def __init__(self, train=True):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3076,)
                ),  # mean, std computed in dataset.ipynb
            ]
        )

        # Download and load the MNIST data
        self.data = datasets.MNIST(
            root=".",
            train=train,
            download=True,
            transform=self.transform,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
