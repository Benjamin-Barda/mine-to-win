from datetime import date
from BackboneDataset import BackboneDataset
import os
import pandas as pd
import numpy as np
import json
import torch
datasets = []

for filename in os.listdir("."):
    
    if filename.startswith("."):
        continue
    
    if len(filename.split(".")) != 2:
        continue

    name, ext = filename.split(".")
    
    if ext == "json" and name.startswith("PARSED"):
        datasets.append(filename)
        
print(datasets)

folders = [ f[13:]  for f in datasets]
folders = [ f[:-5].split("-")[0]  for f in folders]
print(folders)



for js, img in zip(datasets, folders):
    
    d = BackboneDataset("imgs/" + img, js)
    torch.save(d, os.path.join("data", "datasets", js + ".dtset"))

    