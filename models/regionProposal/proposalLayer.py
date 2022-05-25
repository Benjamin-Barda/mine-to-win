# unTODO: Understand if we learn offsets from the feature map or from the original image ???

import os
import sys
import torch.nn as nn
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from utils.config import cfg
from models.regionProposal.utils.anchorUtils import *


class _proposal(nn.Module):

    def __init__(self, training=False):
        super(_proposal, self).__init__()

        self.pre_nms_train = cfg.PRE_NMS_TRAIN,
        self.post_nms_train = cfg.POST_NMS_TRAIN,
        self.pre_nms_test = cfg.PRE_NMS_TEST,
        self.post_nms_test = cfg.POST_NMS_TEST,

    def forward(self, fg_scores, reg_scores, anchors, img_size):
        """
        args :
            fg_score :
            reg_scores
            anchors : {x, y, W, H}
            img_size : {H, W}
        return :
            RoI

        Alg :
            Convert Anchors into proposal ... that is apply offsets from reg to anchor and clip it to image
            Sort them based on fg_scores
            Apply NMS and take only top K
        """

        # Apply predicted offset to original anchors thus turning them into proposals
        # print(anchors.shape)
        rois = getROI(anchors, reg_scores)
        # Let's clip them to the image
        to_clip = centr2corner(rois)

        to_clip[:, :, 0:4:2] = torch.clip(
            to_clip[:, :, 0:4:2], 0, img_size[1]
        )
        to_clip[:, :, 1:4:2] = torch.clip(
            to_clip[:, :, 1:4:2], 0, img_size[0]
        )

        # TODO : Add threshold for too small of anchors
        # TODO : Add PRE and POST nms pruning

        # return the indices of the sorted array reversed
        order = fg_scores.argsort(descending=False)

        # TODO: I have to sort ROIS based on score ... how the fuck i can do IT without this ugly loop ???
        for i in range(len(order)) :
            rois[i] = rois[i, order[i], :]

        print(order.shape)
        print(rois.shape)

        return to_clip


def getROI(src_box, offset):
    # unTODO : Check if the batch size is a problem

    """
    Taken from R-CNN paper

    offset calc  :
        dst_x = src_w * off_x + src_x
        dst_y = src_h * off_y + src_y
        dst_w = src_w * exp(off_w)
        dst_h = src_h * exp(off_h)

    """

    src_x = src_box[:, :, 0::4]
    src_y = src_box[:, :, 1::4]
    src_w = src_box[:, :, 2::4]
    src_h = src_box[:, :, 3::4]

    off_x = offset[:, :, 0::4]
    off_y = offset[:, :, 1::4]
    off_w = offset[:, :, 2::4]
    off_h = offset[:, :, 3::4]

    dst = offset.clone()

    dst[:, :, 0::4] = src_w * off_x + src_x
    dst[:, :, 1::4] = src_h * off_y + src_y
    dst[:, :, 2::4] = src_w * torch.exp(off_w)
    dst[:, :, 3::4] = src_h * torch.exp(off_h)

    return dst
