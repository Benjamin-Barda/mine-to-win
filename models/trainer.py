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

# def l2_norm_loss(anch, p, n, margin):
#     dp = torch.sqrt(torch.pow(anch - p, 2).sum(axis = 1))
#     dn = torch.sqrt(torch.pow(anch - n, 2).sum(axis = 1))
#     return torch.clamp(dp - dn + margin, min=0.0)

# def cos_dist_loss(anch, p, n, margin):
#     dp = 1.0 - torch.diagonal(anch @ p.T) / torch.sqrt(torch.pow(p,2).sum(axis = 1) * torch.pow(anch,2).sum(axis = 1))
#     dn = 1.0 - torch.diagonal(anch @ n.T) / torch.sqrt(torch.pow(n,2).sum(axis = 1) * torch.pow(anch,2).sum(axis = 1))
#     return torch.clamp(dp - dn + margin, min=0.0)

# class TripletLoss(torch.nn.Module):
#     # Shamelessly stolen from https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942
#     def __init__(self, margin, l2_norm = False, mine = False):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#         self.l2_norm = l2_norm
#         self.mine = mine

#     def forward(self, anchor, p, n):
#         if self.mine:
#             if self.l2_norm:
#                 return l2_norm_loss_mine(anchor, p, n, self.margin).sum()
#             else:
#                 return cos_dist_loss_mine(anchor, p, n, self.margin).sum()
#         else:
#             if self.l2_norm:
#                 return l2_norm_loss(anchor, p, n, self.margin).sum()
#             else:
#                 return cos_dist_loss(anchor, p, n, self.margin).sum()


load = False
store = True

dataset = MineDatasetMulti(os.path.join("data", "datasets"), "mine-classes")
loader = TripletLoader(dataset, {'Pig','Cow','Chicken','Sheep','Zombie','Skeleton','Creeper','Spider'}, 10)
loader_iter = iter(loader)
network = BackboneCNN()

if load:
    network.load_state_dict(torch.load("./BackCNN_best_weights.pth"))

optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

best_loss = torch.inf
best_state = network.state_dict()

loss_funct = torch.nn.TripletMarginLoss(margin=10)

i = 0
counter = 0
max_c = 10
while True:
    network.train()
    i += 1
    # Train
    imgs, _ = next(loader_iter)

    preds_a = network.forward(imgs[0][:-10])
    preds_p = network.forward(imgs[1][:-10])
    preds_n = network.forward(imgs[2][:-10])
    loss = loss_funct.forward(preds_a, preds_p, preds_n)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Valid
    network.eval()

    preds_a = network.forward(imgs[0][-10:])
    preds_p = network.forward(imgs[1][-10:])
    preds_n = network.forward(imgs[2][-10:])
    loss = loss_funct.forward(preds_a, preds_p, preds_n)
    
    total_loss = loss.item()
    total_correct = torch.where(loss < 0.01, 1.0, 0.0).sum().item()
    total = 10
    
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
    