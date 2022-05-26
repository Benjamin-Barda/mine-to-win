from concurrent.futures import thread
import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from data.ClassData import ClassData
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Subset
import torch
import cv2
from models.extractor import BackboneCNN
from torch.backends import cuda

# import seaborn as sn
# from sklearn.metrics import confusion_matrix
# import pandas as pd
# import matplotlib.pyplot as plt

# def plot_confusion(lab, pred):
#     cmat = confusion_matrix(lab, pred)
#     df_cm = pd.DataFrame(cmat, index = [i for i in ["Nothing", "Creeper", "Pig"]],
#                     columns = [i for i in ["Nothing", "Creeper", "Pig"]])

#     plt.figure(figsize = (10,7))
#     ax = sn.heatmap(df_cm, annot=True, linewidths=0.5)
#     ax.set_xlabel('Targets', fontsize=14)
#     ax.set_ylabel('Predictions', fontsize=14)
#     plt.show()

cuda.benchmark = True

ds = torch.load("data\\datasets\minedata_classifier_local.dtst")
print(ds.shape())

device = "cuda" if torch.cuda.is_available() else "cpu"

# split = int(ds.shape()[0] * 0.8)
# train, val = Subset(ds, list(range(split))), Subset(ds, list(range(split, ds.shape()[0])))

split = int(ds.shape()[0] * 0.6)
train, val = Subset(ds, list(range(split))), Subset(ds, list(range(split, int(ds.shape()[0] * 0.8))))

BATCH_SIZE = 256

train_load = DataLoader(train, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True)
val_load =   DataLoader(val, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True)

model = BackboneCNN().to(device, non_blocking=True)

load = True
store = True

if load:
    model.load_state_dict(torch.load("./BackCNN_deep3_best_weights2.pth"))

optimizer = torch.optim.AdamW(params=model.parameters(), lr = 0.0002, amsgrad=True)

best_risk = torch.inf
best_state = model.state_dict()

loss_funct = torch.nn.CrossEntropyLoss()

i = 0
max_c = 15

tot_minibatch = np.ceil(split / BATCH_SIZE)
tot_minibatch_val = np.ceil((split * 0.2) / BATCH_SIZE)

while True:

    model.train()
    ds.set_train_mode(True)
    
    for indx, (img, label, _) in enumerate(train_load):
        
        #cv2.imshow("HI", img[0].permute(1,2,0).numpy())
        #cv2.waitKey(-1)
        
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        
        preds = model.forward(img)
        loss = loss_funct.forward(preds, label)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {i}, {(100 * indx / tot_minibatch):.3f}%")
        
    model.eval()
    ds.set_train_mode(False)
    
    with torch.no_grad():
        total_loss, total_correct, total = 0,0,0
        
        for img, label in val_load:
            
            #cv2.imshow("HI", img[0].permute(1,2,0).numpy())
            #cv2.waitKey(-1)
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            total += img.shape[0]
            
            preds = model.forward(img)
            loss = loss_funct.forward(preds, label)

            total_loss += loss.item()
            total_correct += preds.argmax(axis=1).eq(label).sum().item()
            
            
        risk = total_loss / tot_minibatch_val
        accuracy = total_correct / total
        print(f"Epoch {i}: accuracy={accuracy:.5f}, risk={risk:.5f}")
        
        if risk < best_risk:
            best_risk = risk
            best_state = model.state_dict()
            counter = 0
        elif counter < max_c:
            counter += 1
        else:
            print(f"Worse loss reached, stopping training, best risk: {best_risk}.")
            break
    
        i += 1
    if i % 50 == 0:
        torch.save(best_state, f"/content/drive/MyDrive/Colab Notebooks/MineCNN/curr_state{i}_1.pth")

if store:
    torch.save(best_state, "./BackCNN_deep3_best_weights3.pth")