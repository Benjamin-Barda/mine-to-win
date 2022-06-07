import torch

# has to be tested
class RPNLoss(torch.nn.Module):

    def __init__(self, value=10):
        super(RPNLoss, self).__init__()
        self.value = value
        self.cls_loss = torch.nn.BCELoss(reduction='mean') # Instead of dividing the sum by N
        self.reg_loss = torch.nn.SmoothL1Loss(reduction='none') # Same

    def forward(self, classifier, reg, ground_cls, ground_reg):
        # Implemented as Faster R-CNN intended

        ao = ground_cls[:, None].expand(-1, 4) * self.reg_loss(reg, ground_reg)

        reg_loss = self.value * ao.sum() / ao.shape[0]
        class_loss = self.cls_loss(classifier, ground_cls)


        return reg_loss + class_loss