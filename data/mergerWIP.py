from datetime import date
from BackboneDataset import BackboneDataset
import os
import pandas as pd
import numpy as np
import json

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
folders = [ f[:-5]  for f in folders]
print(folders)

df = pd.DataFrame

for js, img in zip(datasets, folders):
    
    d = BackboneDataset("imgs/" + img, js)
    pd.concat([df, d.data], ignore_index=True, axis=0)
    
print(len(df))
    
    