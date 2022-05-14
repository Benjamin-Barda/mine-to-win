from datetime import date
from BackboneDataset import BackboneDatasetMulti
import os
import pandas as pd
import numpy as np
import json
import torch
datasets = []

for filename in os.listdir("jsons/"):
    
    if filename.startswith("."):
        continue
    
    if len(filename.split(".")) != 2:
        continue

    name, ext = filename.split(".")
    
    if ext == "json" and name.startswith("PARSED"):
        datasets.append(filename)
        

folders = [ f[13:]  for f in datasets]
folders = [ f[:-5].split("-")[0]  for f in folders]

folders = ["imgs\\" + img for img in folders]


datasets = ["jsons\\" + dataset for dataset in datasets]

print(datasets)
print(folders)
    
d = BackboneDatasetMulti(folders, datasets)

print(d.data.shape)
torch.save(d, os.path.join("data", "datasets", "mine-win.dtset"))

    