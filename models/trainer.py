import torch
from backCNN import BackboneCNN
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../data"))
from TripletLoader import TripletLoader
from MineDataset import MineDatasetMulti

# def cos_dist_loss_mine(anch, p, n, margin):
#     dp = 1.0 - torch.diagonal(anch @ p.T) / torch.sqrt(torch.pow(p,2).sum(axis = 1) * torch.pow(anch,2).sum(axis = 1))
#     dn = 1.0 - torch.diagonal(anch @ n.T) / torch.sqrt(torch.pow(n,2).sum(axis = 1) * torch.pow(anch,2).sum(axis = 1))
#     return (dp + margin) / dn

# def l2_norm_loss_mine(anch, p, n, margin):
#     dp = torch.sqrt(torch.pow(anch - p, 2).sum(axis = 1))
#     dn = torch.sqrt(torch.pow(anch - n, 2).sum(axis = 1))
#     return (dp + margin) / dn

def l2_norm_loss(anch, p, n, margin):
    dp = torch.sqrt(torch.pow(anch - p, 2).sum(axis = (1,2,3)))
    dn = torch.sqrt(torch.pow(anch - n, 2).sum(axis = (1,2,3)))
    return torch.clamp(dp - dn + margin, min=0.0)

def cos_dist_loss(anch, p, n, margin):
    dp = 1.0 - torch.diag(torch.tensordot(anch,p, dims=([1,2,3],[1,2,3]))) / torch.sqrt(torch.pow(p,2).sum(axis = (1,2,3)) * torch.pow(anch,2).sum(axis = (1,2,3)))
    dn = 1.0 - torch.diag(torch.tensordot(anch,n, dims=([1,2,3],[1,2,3]))) / torch.sqrt(torch.pow(n,2).sum(axis = (1,2,3)) * torch.pow(anch,2).sum(axis = (1,2,3)))
    return torch.clamp(dp - dn + margin, min=0.0)

class TripletLoss(torch.nn.Module):
    # Shamelessly stolen from https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, p, n):
        return cos_dist_loss(anchor, p, n, self.margin).sum()


load = False
store = True

# define the device for the computation
device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

BATCH_SIZE = 2
VALID_SIZE = 1
    
dataset = MineDatasetMulti(os.path.join("data", "datasets"), "mine-classes")
loader = TripletLoader(dataset, {'Pig','Cow','Chicken','Sheep','Zombie','Skeleton','Creeper','Spider'}, BATCH_SIZE)
loader_iter = iter(loader)
network = BackboneCNN().to(device)
torch.autograd.set_detect_anomaly(True)

import cv2
    
if load:
    network.load_state_dict(torch.load("./BackCNN_best_weights.pth"))

optimizer = torch.optim.AdamW(network.parameters(), lr=0.001)

best_loss = torch.inf
best_state = network.state_dict()

loss_funct = TripletLoss(margin=0.1)

i = 0
counter = 0
max_c = 10
MAX_ITERS = 5

#3, 1, 3, 360, 640
valid = torch.zeros((3, MAX_ITERS, 3, 360, 640))

while True:
    i += 1
    j = 0
    
    valid = valid.permute(1, 0, 2, 3, 4).cpu()

    network.train()
    
    while j < MAX_ITERS:
        imgs, _ = next(loader_iter)
    
        # Train

        imgs = imgs.to(device=device)
        preds_a = network.forward(imgs[0][:-VALID_SIZE])
        preds_p = network.forward(imgs[1][:-VALID_SIZE])
        preds_n = network.forward(imgs[2][:-VALID_SIZE])
    
        loss = loss_funct.forward(preds_a, preds_p, preds_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        valid[j] = imgs.permute(1, 0, 2, 3, 4)[-VALID_SIZE:].cpu()
            
        j += 1


    # Valid
    network.eval()
    with torch.no_grad():
        
        valid = valid.permute(1, 0, 2, 3, 4).to(device=device)
        preds_a = network.forward(valid[0])
        preds_p = network.forward(valid[1])
        preds_n = network.forward(valid[2])

        loss = loss_funct.forward(preds_a, preds_p, preds_n)
    
        total_loss = loss.item()
        total_correct = torch.where(cos_dist_loss(preds_a, preds_p, preds_n, 0) == 0, 1.0, 0.0).sum().item()
        total = preds_a.size(1)
        
        risk = total_loss / total
        accuracy = total_correct / total
        print(f"Epoch {i}: accuracy={accuracy:.5f}, risk={risk:.5f}")

        if loss < best_loss:
            best_loss = loss
            best_state = network.state_dict()
            counter = 0
        elif counter < max_c:
            counter += 1
        else:
            print(f"Worse loss reached, stopping training...")
            break


if store:
    torch.save(best_state, "./BackCNN_best_weights.pth")
    