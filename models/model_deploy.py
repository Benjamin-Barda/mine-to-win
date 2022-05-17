from msilib.schema import Class
import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../data"))
from ClassLoader import ClassData
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import torch
import cv2
from backCNN import BackboneCNN
from torch.backends import cuda

trans = transforms.ToTensor()

data = ClassData(["jsons\\PARSEDOUTPUT-2022-05.json"], transform = trans)
dl = DataLoader(data, pin_memory=True, batch_size=2)

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

model = BackboneCNN().to("cuda")
model.load_state_dict(torch.load("./BackCNN_best_weights.pth"))
lossfn = torch.nn.CrossEntropyLoss()
model.eval()

model.conv5.register_forward_hook(get_features('feats'))

cv2.namedWindow("Sample", cv2.WINDOW_NORMAL)
cv2.namedWindow("Map", cv2.WINDOW_NORMAL)

PREDS = []
FEATS = []
features = {}

for img, label in dl:
    
    img, label = img.to("cuda"), label.to("cuda")
    preds = model.forward(img)

    for indx, im in enumerate(img):
        cv2.imshow("Sample", im.to("cpu").permute(1,2,0).numpy())
        x = features['feats']
        cv2.imshow("Map", features['feats'][indx].permute(1,2,0).cpu().numpy())
        cv2.waitKey(-1)
        
    
    label = label.to("cpu")
    print(preds.argmax(axis=1), label)
    
cv2.destroyAllWindows()