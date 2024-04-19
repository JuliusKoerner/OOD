import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Datasets import load
import yaml
from model import Conv_Net, Conv_Net_Dropout
import sys
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def load_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


# Main function to setup and run training
def run(config):
    basepath = config["logs"]
    writer = SummaryWriter(f"{basepath}/runs_{len(os.listdir(basepath))}")

    train_dataset = load("MNIST")(train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_dataset = load("MNIST")(train=False)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    # Model, Loss, and Optimizer
    model = Conv_Net_Dropout()
    model = model.to("cuda")
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training loop
    model.train()
    min_val_loss = 0
    for epoch in tqdm(range(config["epochs"])):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.to("cuda"))
            loss = loss_func(output, target.to("cuda"))
            loss.backward()
            optimizer.step()
            writer.add_scalar(
                "Loss/train", loss.item(), epoch * len(train_loader) + batch_idx
            )

        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data.to("cuda"))
                val_loss += loss_func(output, target.to("cuda")).item()
            val_loss /= len(val_loader)
            writer.add_scalar("Loss/val", val_loss, epoch)
        if val_loss < min_val_loss or min_val_loss == 0:
            min_val_loss = val_loss
            torch.save(model.state_dict(), config["store"])
            print("Model saved")
    writer.add_hparams(config, {"min_val_loss": min_val_loss})


if __name__ == "__main__":
    config = load_config(sys.argv[1])
    run(config)
