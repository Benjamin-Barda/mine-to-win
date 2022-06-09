import os
import sys
import torch
import torch.nn as nn
from torchvision.ops import nms


sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from utils.config import cfg
from models.regionProposal.utils.anchorUtils import *


class _proposal(nn.Module):

    def __init__(self, training=False, device="cpu"):
        super(_proposal, self).__init__()

        self.device = device
        self.pre_nms_train = cfg.PRE_NMS_TRAIN
        self.post_nms_train = cfg.POST_NMS_TRAIN
        self.pre_nms_test = cfg.PRE_NMS_TEST
        self.post_nms_test = cfg.POST_NMS_TEST
        self.nms_thresh = cfg.NMS_THRESHOLD

        self.is_training = training

    def forward(self, scores, rois, img_size):

        batch_size = rois.shape[0]

        post_nms = self.post_nms_train if self.is_training else self.post_nms_test

        # Apply predicted offset to original anchors thus turning them into proposals

        # rois = self.getROI(anchors, reg_scores)
        # Let's clip them to the image
        to_clip = center2corner(rois)

        to_clip = torch.clamp(
            to_clip, min=torch.zeros(4, device=self.device), max=torch.tensor([img_size[1], img_size[0], img_size[1], img_size[0]], device=self.device) - 1
        )

        # BENJAMIN I DEMAND YOUR HEAD ON A F**#@* PIKE
        #   I WENT TO SLEEP AT 2 am FOR THIS S*?#
        # # # return the indices of the sorted array reversed
        # # # order = scores.argsort(dim=1, descending=True)

        # # # # Sort region of interest based on score
        # # # for i in range(batch_size):
        # # #     to_clip[i] = to_clip[i, order[i]]
        # # #     scores[i] = scores[i, order[i]]

        indexes = list()

        for i in range(batch_size):

            to_keep = nms(
                to_clip[i],
                scores[i],
                self.nms_thresh
                )

            if post_nms > 0:
                to_keep = to_keep[:post_nms]
            
            indexes.append(to_keep)

        return to_clip, indexes

    def getROI(self, src_box, offset):
        # unTODO : This is useless!

        src_x = src_box[:, :, 0::4]
        src_y = src_box[:, :, 1::4]
        src_w = src_box[:, :, 2::4]
        src_h = src_box[:, :, 3::4]

        off_x = offset[:, :, 0::4]
        off_y = offset[:, :, 1::4]
        off_w = offset[:, :, 2::4]
        off_h = offset[:, :, 3::4]

        dst = offset.clone().to(self.device)

        dst[:, :, 0::4] = src_w * off_x + src_x
        dst[:, :, 1::4] = src_h * off_y + src_y
        dst[:, :, 2::4] = src_w * torch.exp(off_w)
        dst[:, :, 3::4] = src_h * torch.exp(off_h)

        return dst
