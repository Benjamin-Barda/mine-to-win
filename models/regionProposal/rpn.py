import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from torch import nn
from torch.nn import functional as F
from models.regionProposal.proposalLayer import _proposal
from utils.config import cfg
from models.regionProposal.utils.anchorUtils import *


class _rpn(nn.Module):

    def __init__(self, inDimension, feature_stride=16):
        super(_rpn, self).__init__()

        # Depth of the in feature map
        self.inDimension = inDimension
        self.feature_stride = feature_stride

        self.baseConvOut = cfg.BASE_CONV_OUT_SIZE

        # Scales and ratio of the anchors
        self.anchorScales = cfg.ANCHOR_SCALES
        self.anchorRatios = cfg.ANCHOR_RATIOS

        self.A = len(self.anchorScales) * len(self.anchorRatios)

        self.anchors = generate_anchors(1, self.anchorRatios, self.anchorScales)

        # Base of the convolution
        self.BASE_CONV = nn.Sequential(
            nn.Conv2d(self.inDimension, self.baseConvOut, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True))

        # -> Region Proposal LAyer here

        # Classification layer
        self.cls_out_size = len(self.anchorScales) * len(self.anchorRatios) * 2
        self.classificationLayer = nn.Conv2d(self.baseConvOut, self.cls_out_size, kernel_size=1, stride=1, padding=0)

        # Regression Layer on the BBOX
        self.regr_out_size = 4 * self.A
        self.regressionLayer = nn.Conv2d(self.baseConvOut, self.regr_out_size, kernel_size=1, stride=1, padding=0)

        self.proposalLayer = _proposal()

    def forward(self, x, img_size):
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
        anchors = self.splashAnchors(fH, fW, n)

        # Pass BASE first into the regressor -> BBox offset and scales for anchors
        rpn_reg = self.regressionLayer(base)

        # (n, W * H * A, 4)
        rpn_reg = rpn_reg.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # Pass BASE into the classificator -> Fg / Bg scores
        rpn_score = self.classificationLayer(base)
        rpn_score = rpn_score.permute(0, 2, 3, 1).contiguous()

        # The paper suggest a 2 class softmax architechture for the classification layer
        rpn_softmax = F.softmax(rpn_score.view(n, fH, fW, self.A, 2), dim=4)
        # take only the foreground prediction 
        fg_scores = rpn_softmax[:, :, :, :, 1].contiguous()
        fg_scores = fg_scores.view(n, -1)
        rpn_score = rpn_score.view(n, -1, 2)

        # At the end we have 
        #   rpn_score.shape   = (n, W*H*A, 2) 
        #   rpn_reg.shape     = (n, W*H*A, 4)
        #   rpn_softmax.shape = (n*H*W*A, 1)

        rois = self.proposalLayer(
            fg_scores,
            rpn_reg,
            anchors,
            img_size
        )

        return rpn_score, rpn_reg, rois

    def splashAnchors(self, feat_height, feat_width, batch_size):
        shift_center_x = torch.arange(0, feat_width * self.feature_stride, self.feature_stride)
        shift_center_y = torch.arange(0, feat_height * self.feature_stride, self.feature_stride)

        shift_center_x, shift_center_y = np.meshgrid(shift_center_x, shift_center_y)
        shift_center_x = shift_center_x.ravel()
        shift_center_y = shift_center_y.ravel()
        # TODO: Height and width of the anchors are not modified ... this is beacuase regression is done in the image
        #  space - Question is if it is correct ????
        shifts = np.stack(
            (shift_center_x, shift_center_y,
             np.zeros(shift_center_x.shape[0]), np.zeros(shift_center_y.shape[0])), axis=1)

        K = shifts.shape[0]

        anchor = self.anchors.reshape((1, self.A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchor = anchor.view(K * self.A, 4).expand(batch_size, K * self.A, 4)

        return anchor
