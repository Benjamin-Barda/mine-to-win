from MineDataset import create_dataset
import os
datasets = []

for filename in os.listdir("jsons/"):
    
    if filename.startswith("."):
        continue
    
    if len(filename.split(".")) != 2:
        continue

    name, ext = filename.split(".")
    
    if ext == "json" and name.startswith("PARSED"):
        datasets.append(filename)

datasets = ["jsons\\" + dataset for dataset in datasets]

print(datasets)
    
create_dataset(datasets, "imgs\\frames\\", os.path.join("data", "datasets"), "mine-classes")