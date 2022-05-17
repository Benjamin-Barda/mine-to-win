import MineDataset
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

datasets = [os.path.join("jsons", dataset) for dataset in datasets]
    
MineDataset.create_dataset_with_tensor(datasets, os.path.join("imgs","frames"), os.path.join("data", "datasets"), "mine-classes")