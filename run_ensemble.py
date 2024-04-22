import src.train_model as train_model
import yaml

runs = 10
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

for i in range(10, runs + 10):
    config["store"] = f"models/ensemble_10ep/run_{i}.pth"
    train_model.run(config)
