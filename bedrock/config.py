import os
import yaml

file_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(file_dir, "config.yml")

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

