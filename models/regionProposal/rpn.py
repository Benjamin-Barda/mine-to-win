import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from torch import nn
from torch.nn import functional as F
from models.regionProposal.proposalLayer import _proposal
from utils.config import cfg
from models.regionProposal.utils.anchorUtils import *


class _rpn(nn.Module):

    def __init__(self, inDimension, feature_stride=cfg.FEATURE_STRIDE, device = "cpu"):
        super(_rpn, self).__init__()
        
        self.device = device

        # Depth of the in feature map
        self.inDimension = inDimension
        self.feature_stride = feature_stride

        self.baseConvOut = cfg.BASE_CONV_OUT_SIZE

        # Scales and ratio of the anchors
        self.anchorScales = cfg.ANCHOR_SCALES
        self.anchorRatios = cfg.ANCHOR_RATIOS

        self.A = len(self.anchorScales) * len(self.anchorRatios)

        self.anchors = generate_anchors(40, self.anchorRatios, self.anchorScales)

        # Base of the convolution
        self.BASE_CONV = nn.Sequential(
            nn.Conv2d(self.inDimension, self.baseConvOut, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Mish(inplace=True),
            nn.Dropout2d(p = 0.2, inplace=True),
            nn.BatchNorm2d(self.baseConvOut)
            )

        # -> Region Proposal Layer here

        # Classification layer
        self.cls_out_size = 2 * self.A
        self.classificationLayer = nn.Sequential(
            nn.Conv2d(self.baseConvOut, self.cls_out_size, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.cls_out_size)
        )
        # Regression Layer on the BBOX
        self.regr_out_size = 4 * self.A
        self.regressionLayer = nn.Sequential(
            nn.Conv2d(self.baseConvOut, self.regr_out_size, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.regr_out_size)
        )
        self.proposalLayer = _proposal(device=self.device)

        nn.init.kaiming_uniform_(self.BASE_CONV[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.classificationLayer[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.regressionLayer[0].weight, nonlinearity='relu')


    def forward(self, x , img_size, training=True):
        '''
        args : 
            x : tensor : Feature map give
            n by the last convolutional layer of the backbone network
        '''

        # n  : Batchsize
        # c  : number of channels
        # fH : Feature map heigth
        # fW : Feature map width 
        n, c, fH, fW = x.shape

        # Pass into first conv layer + ReLU
        base = self.BASE_CONV(x)
        anchors = splashAnchors(fH, fW, n, self.anchors, img_size, self.feature_stride, A=self.A, training=training)

        anchors = anchors.to(self.device, non_blocking=True)

        # Pass BASE first into the regressor -> BBox offset and scales for anchors
        rpn_reg = self.regressionLayer(base)

        # (n, W * H * A, 4)
        rpn_reg = rpn_reg.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # Pass BASE into the classificator -> Fg / Bg scores
        rpn_score = self.classificationLayer(base)
        rpn_score = rpn_score.permute(0, 2, 3, 1).contiguous()

        # The paper suggest a 2 class softmax architecture for the classification layer
        rpn_softmax = F.softmax(rpn_score.view(n, fH, fW, self.A, 2), dim=4)
        # take only the foreground prediction 
        fg_scores = rpn_softmax[:, :, :, :, 1].contiguous()
        fg_scores = fg_scores.view(n, -1)
        #rpn_score = rpn_score.view(n, -1, 2)

        # At the end we have 
        #   rpn_score.shape   = (n, W*H*A, 2) 
        #   rpn_reg.shape     = (n, W*H*A, 4)
        #   rpn_softmax.shape = (n*H*W*A, 1)

        rpn_reg = rpn_reg + anchors

        rois = self.proposalLayer(
            fg_scores,
            rpn_reg,
            anchors,
            img_size
        )

        rois = corner2center(rois)

        #print(f"anchor:  {anchors.shape}\nrois : {rois.shape}")

        ts = torch.empty((n, rpn_reg.shape[1], 4), device=cfg.DEVICE, dtype = torch.float)

        if torch.where(rois[:, :, 2:] < 0, 1, 0).sum() > 0:
            print(torch.min(rois[:, :, 2:]))

        ts[:, :, 0] = (rois[:, :, 0] - anchors[:, :, 0]) / anchors[:, :, 2]
        ts[:, :, 1] = (rois[:, :, 1] - anchors[:, :, 1]) / anchors[:, :, 3]
        ts[:, :, 2] = torch.log(torch.clamp(rois[:, :, 2]/ anchors[:, :, 2], min=1e-300))
        ts[:, :, 3] = torch.log(torch.clamp(rois[:, :, 3]/ anchors[:, :, 3], min=1e-300))

        if torch.isnan(ts[:, :, 0]).sum() > 0:
            print("tx nan")
        if torch.isnan(ts[:, :, 1]).sum() > 0:
            print("ty nan")
        if torch.isnan(ts[:, :, 2]).sum() > 0:
            print("tw nan")
        if torch.isnan(ts[:, :, 3]).sum() > 0:
            print("th nan")

        return fg_scores, ts, rois
