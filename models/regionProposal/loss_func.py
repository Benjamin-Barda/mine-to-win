import torch

# has to be tested
class RPNLoss(torch.nn.Module):
    def __init__(self, value=10):
        super(RPNLoss, self).__init__()
        self.value = value
        self.cls_loss = torch.nn.NLLLoss(reduction='mean') # Instead of dividing the sum by N
        self.reg_loss = torch.nn.SmoothL1Loss(reduction='mean') # Same

    def forward(self, cls, reg, ground_cls, ground_reg):
        # Implemented as Faster R-CNN intended
        return self.cls_loss(cls, ground_cls) + self.value * self.reg_loss(reg, ground_reg)
