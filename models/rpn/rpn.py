import sys
import os
from turtle import forward
import torch
from torch import nn 
from torch.nn import functional as F 
import pandas as pd
import numpy as np
from utils.anchorUtils import generate_anchors, splashAnchors
from proposalLayer import _proposal

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from models.utils.config import cfg

class rpn (nn.Module) : 

    def __init__(self, inDimension, feature_stride = 15): 

        super(rpn, self).__init__()

        # Depth of the in feature map
        self.inDimension = inDimension
        self.feature_stride = feature_stride

        self.baseConvOut = cfg.BASE_CONV_OUT_SIZE

        # Scales and ratio of the anchors
        self.anchorScales = cfg.ANCHOR_SCALES
        self.anchorRatios = cfg.ANCHOR_RATIOS

        self.A = self.anchorRatios.shape[0] * self.anchorScales.shape[0]

        self.anchors = generate_anchors(16, self.anchorRatios, self.anchorScales)


        # Base of the convolution
        self.BASE_CONV = nn.Sequential(
            nn.Conv2d(self.inDimension, self.baseConvOut, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True))

        # -> Region Proposal LAyer here

        # Classification layer
        self.cls_out_size = len(self.anchorScales) * len(self.anchorRatios) * 2 
        self.classificationLayer = nn.Conv2d(self.baseConvOut, self.cls_out_size, kernel_size= 1, stride = 1, padding=0 )

        # Regression Layer on the BBOX
        self.regr_out_size = 4 * len(self.anchorScales) * len(self.anchorRatios)
        self.regressionLayer = nn.Conv2d(self.baseConvOut, self.regr_out_size, kernel_size=1, stride=1, padding=0)

        self.proposalLayer = _proposal()


    def forward(self, x, img_size) : 
        
        '''
        args : 
            x : tensor : Feature map given by the last convolutional layer of the backbone network
        '''

        # n  : Batchsize
        # c  : number of channels
        # fH : Feature map heigth
        # fW : Feature map width 
        n, c, fH, fW = x.shape

        # Pass into first conv layer + ReLU
        base = self.BASE_CONV(x) 

        anchors = splashAnchors(self.anchors)

        # Pass BASE first into the regressor -> BBox offset and scales for anchors
        rpn_reg = self.regressionLayer(base)

        # (n, W * H * A, 4)
        rpn_reg = rpn_reg.permute(0,2,3,1).view(n, -1, 4)

        # Pass BASE into the classificator -> Fg / Bg scores
        rpn_score = self.classificationLayer(base)
        rpn_score = rpn_score.permute(0,2,3,1)

        # The paper suggest a 2 class softmax architechture for the classification layer
        rpn_softmax = F.softmax(rpn_score.view(n, fH, fW, self.A, 2 ), dim = 4)            
        # take only the foreground prediction 
        fg_scores = rpn_softmax[:, :, :, :, 1]
        fg_scores = fg_scores.view(n, 1)
        rpn_score = rpn_score.view(n, -1, 2)

        # At the end we have 
        #   rpn_score.shape   = (n, W*H*A, 2) 
        #   rpn_reg.shape     = (n, W*H*A, 4)
        #   rpn_softmax.shape = (n*H*W*A, 1)


        rois = list()
        rois_indxs = list()
        for batch_index in range(n) :

            roi = self.proposalLayer(
                fg_scores[batch_index].data.numpy(),
                rpn_reg[batch_index].data.numpy(), 
                anchors, 
                img_size 
                )
            rois.append(roi)
            rois_indxs.append(rois_indxs)
        
        return rpn_score, rpn_reg, rois, rois_indxs

        






      
        



        
        

