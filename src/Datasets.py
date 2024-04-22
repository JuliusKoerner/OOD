import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


def load(name):
    datasets = {"MNIST": MNISTDataset, "CIFAR10": CIFAR10Dataset}
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
            transform=self.transform,  # normalize and convert to tensor
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CIFAR10Dataset(Dataset):
    def __init__(self, train=True):
        """
        Initializes the CIFAR-10 dataset with specified transformations.
        """
        self.transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),  # Resize the images to 28x28
                transforms.Grayscale(
                    num_output_channels=1
                ),  # Convert images to grayscale
                transforms.ToTensor(),
                transforms.Normalize((0.4811,), (0.2311,)),
            ]
        )

        # Download and load the CIFAR-10 data
        self.data = datasets.CIFAR10(
            root="CIFAR10",
            train=train,
            download=True,
            transform=self.transform,  # Apply the composed transformations
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
