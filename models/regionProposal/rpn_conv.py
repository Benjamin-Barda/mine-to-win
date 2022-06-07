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
        self.anchors.requires_grad = False

        self.BASE_CONV = nn.Sequential(
            nn.Conv2d(self.inDimension, self.baseConvOut, kernel_size=3, padding='same'),
            nn.Mish(inplace=True),
            nn.Dropout2d(p = 0.2, inplace=True),
            nn.BatchNorm2d(self.baseConvOut),
        )

        # -> Region Proposal Layer here

        # Classification layer
        self.classificationLayer = nn.Sequential(
            nn.Conv2d(self.baseConvOut, self.A, kernel_size=5, padding='same', bias=False),
            nn.BatchNorm2d(self.A),
            nn.Sigmoid()
        )
        # Regression Layer on the BBOX
        self.regr_out_size = 4 * self.A
        self.regressionLayer = nn.Sequential(
            nn.Conv2d(self.baseConvOut, self.regr_out_size, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(self.regr_out_size)
        )
        self.proposalLayer = _proposal(device=self.device)


    def forward(self, x , img_size, training=False):
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

        anchors = splashAnchors(fH, fW, n, self.anchors, img_size, self.feature_stride, A=self.A, training=training)

        anchors.requires_grad = False
        anchors = anchors.to(self.device, non_blocking=True)

        base = self.BASE_CONV(x)

        # Pass BASE first into the regressor -> BBox offset and scales for anchors
        rpn_reg = self.regressionLayer(base)

        # (n, W * H * A, 4)
        rpn_reg = rpn_reg.view(n, rpn_reg.shape[1] * rpn_reg.shape[2] * rpn_reg.shape[3] // 4, 4)
        # Pass BASE into the classificator -> Fg / Bg scores
        rpn_score = self.classificationLayer(base)
        rpn_score = rpn_score.view(n, rpn_score.shape[1] * rpn_score.shape[2] * rpn_score.shape[3])

        # At the end we have 
        #   rpn_score.shape   = (n, W*H*A) 
        #   rpn_reg.shape     = (n, W*H*A, 4)

        rpn_reg = rpn_reg + anchors

        rois, nms_indexes = self.proposalLayer(
            rpn_score,
            rpn_reg,
            img_size
        )

        rois = corner2center(rois)

        #print(f"anchor:  {anchors.shape}\nrois : {rois.shape}")
        ts = torch.zeros_like(rpn_reg, device=cfg.DEVICE, dtype = torch.float32)

        ts[:, :, 0] = (rois[:, :, 0] - anchors[:, :, 0]) / anchors[:, :, 2]
        ts[:, :, 1] = (rois[:, :, 1] - anchors[:, :, 1]) / anchors[:, :, 3]
        ts[:, :, 2] = torch.log(torch.clamp(rois[:, :, 2]/ anchors[:, :, 2], min=1e-20))
        ts[:, :, 3] = torch.log(torch.clamp(rois[:, :, 3]/ anchors[:, :, 3], min=1e-20))

        return rpn_score, ts, rois, nms_indexes
