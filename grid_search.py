import train_model
import yaml

batch_sizes = [32, 64, 128]
learning_rates = [0.001, 0.01, 0.1]

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        config["batch_size"] = batch_size
        config["learning_rate"] = learning_rate
        config["store"] = "models/run_{i}.pth"
        train_model.run(config)
