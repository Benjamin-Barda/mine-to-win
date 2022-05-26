import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from utils.config import cfg


def generate_anchors(base, ratios, scales):
    """
    args :
        base   : int : W and H of windows
        ratios : array : ratios to apply to base
        scales : array : scales to apply to base
    return :
        anchors Tensor : (A,4) {x_center, y_center, Width, height}
    """

    return torch.from_numpy(np.vstack(
        [[0, 0, base * scale * np.sqrt(ratio), base * scale * np.sqrt(1. / ratio)] for scale in scales for ratio in
         ratios]
    ))


def center2corner(tensor_batch):
    n_batch, n_box, _ = tensor_batch.shape

    x = tensor_batch[:, :, 0::4]
    y = tensor_batch[:, :, 1::4]
    w = tensor_batch[:, :, 2::4]
    h = tensor_batch[:, :, 3::4]

    trans = tensor_batch.clone()

    trans[:, :, 0::4] = x - (w / 2)
    trans[:, :, 1::4] = y - (h / 2)
    trans[:, :, 2::4] = x + (w / 2)
    trans[:, :, 3::4] = y + (h / 2)

    return trans


def corner2center(tensor_batch):
    n_batch, n_box, _ = tensor_batch.shape

    x0 = tensor_batch[:, :, 0::4]
    y0 = tensor_batch[:, :, 1::4]
    x1 = tensor_batch[:, :, 2::4]
    y1 = tensor_batch[:, :, 3::4]

    trans = tensor_batch.clone()

    # x_ctr
    trans[:, :, 0::4] = (x1 - x0) / 2
    # y_ctr
    trans[:, :, 1::4] = (y1 - y0) / 2
    # Width
    trans[:, :, 2::4] = x1 - x0
    # Height
    trans[:, :, 3::4] = y1 - y0

def splashAnchors(feat_height, feat_width, batch_size, base_anchors, feature_stride=cfg.FEATURE_STRIDE, A=cfg.A):
    shift_center_x = torch.arange(0, feat_width  * feature_stride, feature_stride)
    shift_center_y = torch.arange(0, feat_height * feature_stride, feature_stride)
    shift_center_x, shift_center_y = np.meshgrid(shift_center_x, shift_center_y)
    shift_center_x = shift_center_x.ravel()
    shift_center_y = shift_center_y.ravel()

    # TODO: Height and width of the anchors are not modified ... this is beacuase regression is done in the image
    #  space - Question is if it is correct ????
    shifts = np.stack(
        (shift_center_x, shift_center_y, 
            np.zeros(shift_center_x.shape[0]), np.zeros(shift_center_y.shape[0])), axis=1)

    K = shifts.shape[0]

    anchor = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.view(K * A, 4).expand(batch_size, K * A, 4)

    return anchor

def label_anchors(image_info, feat_height, feat_width, base_anchors, feature_stride=cfg.FEATURE_STRIDE, A=cfg.A):

    sp_anch = splashAnchors(feat_height, feat_width, 1, base_anchors, feature_stride, A=A)[0].T
    labels = torch.zeros(len(image_info), sp_anch.shape[1])
    values = torch.zeros(len(image_info), 4, sp_anch.shape[1])
    # 4 * n_anchors
    for indx, infos in enumerate(image_info):
        boxes = torch.Tensor([box for boxes_of_label in infos for box in boxes_of_label]) # 4 * n_boxes
        if boxes.shape[0] > 0:
            boxes = boxes.T
            sp_anch_mesh = torch.broadcast_to(sp_anch[:, :, None], (4, sp_anch.shape[1], boxes.shape[1]))
            boxes_mesh = torch.broadcast_to(boxes[:, None, :], (4, sp_anch.shape[1], boxes.shape[1]))
            # x, y, h, w


            # Calculate intersections and IoU
            w_i = torch.clip((sp_anch_mesh[3] + boxes_mesh[3]) / 2 - torch.abs(boxes[0] - sp_anch_mesh[0]), min=0)
            h_i = torch.clip((sp_anch_mesh[2] + boxes_mesh[2]) / 2 - torch.abs(boxes[1] - sp_anch_mesh[1]), min=0)
            I = w_i * h_i
            U = boxes_mesh[3] * boxes_mesh[2] + sp_anch_mesh[3] * sp_anch_mesh[2] - I
            IoU = I / U # n_anchors * n_boxes

            max_iou, max_indices = torch.max(IoU, dim=1) # Why yes, we do really need the indices

            # Classification object or not
            labels[indx] = torch.where(max_iou <= .3, -1, 0) + torch.where(max_iou >= .7, 1, 0)
            # -1 is negative, 0 is null and 1 is positive

            # Values for regressor
            boxes = boxes[:, max_indices]
            values[indx, 0] = (boxes[0] - sp_anch[0])/sp_anch[3]
            values[indx, 1] = (boxes[1] - sp_anch[1])/sp_anch[2]
            values[indx, 2] = torch.log(boxes[2]/sp_anch[2])
            values[indx, 3] = torch.log(boxes[3]/sp_anch[3])

        else:
            labels[indx] = -torch.ones(sp_anch.shape[1])
    
    return labels, values
