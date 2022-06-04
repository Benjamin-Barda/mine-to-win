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

    return torch.tensor([[0., 0., base * scale * np.sqrt(ratio), base * scale * np.sqrt(1. / ratio)] 
                        for scale in scales 
                        for ratio in ratios])


@torch.jit.script
def center2corner(tensor_batch):
    x = tensor_batch[:, :, 0]
    y = tensor_batch[:, :, 1]
    w = .5 * tensor_batch[:, :, 2].abs()
    h = .5 * tensor_batch[:, :, 3].abs()

    trans = torch.empty_like(tensor_batch, device=tensor_batch.device)

    trans[:, :, 0] = x - w
    trans[:, :, 1] = y - h
    trans[:, :, 2] = x + w
    trans[:, :, 3] = y + h

    return trans

@torch.jit.script
def corner2center(tensor_batch):

    x0 = tensor_batch[:, :, 0]
    y0 = tensor_batch[:, :, 1]
    x1 = tensor_batch[:, :, 2]
    y1 = tensor_batch[:, :, 3]

    trans = torch.empty_like(tensor_batch, device=tensor_batch.device)

    # x_ctr
    trans[:, :, 0] = (x1 + x0) / 2
    # y_ctr
    trans[:, :, 1] = (y1 + y0) / 2
    # Width
    trans[:, :, 2] = x1 - x0
    # Height
    trans[:, :, 3] = y1 - y0

    return trans


def splashAnchors(feat_height, feat_width, batch_size, base_anchors, im_size, feature_stride=cfg.FEATURE_STRIDE, A=cfg.A, device = cfg.DEVICE, training = False):
                
    shift_center_x = torch.arange(0, feat_width  * feature_stride, feature_stride)
    shift_center_y = torch.arange(0, feat_height * feature_stride, feature_stride)
    shift_center_x, shift_center_y = torch.meshgrid(shift_center_x, shift_center_y, indexing='ij')
    shift_center_x = shift_center_x.ravel()
    shift_center_y = shift_center_y.ravel()

    # TODO: Height and width of the anchors are not modified ... this is beacuase regression is done in the image
    #  space - Question is if it is correct ????
    shifts = torch.stack((
            shift_center_x,
            shift_center_y, 
            torch.zeros(shift_center_x.shape[0]), 
            torch.zeros(shift_center_y.shape[0])), axis=1)

    K = shifts.shape[0]
    anchor = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).permute((1, 0, 2))
    anchor = anchor.view(K * A, 4)

    if training : 
        H, W = im_size

        keep = ((anchor[ :, 0] - (anchor[ :,2] / 2) >= 0 ) &
            (anchor[ :, 0] + (anchor[ :,2] / 2) <= W ) & 
            (anchor[ :, 1] - (anchor[ :,3] / 2) >= 0 ) & 
            (anchor[ :, 1] + (anchor[ :,3] / 2) <= H ))
        
        inside = torch.nonzero(keep).view(-1)
        anchor = anchor[inside ,: ]

        K = anchor.shape[0]
        return anchor.expand(batch_size, K, 4)
    else: 
        return anchor.expand(batch_size, K*A, 4)
            
@torch.jit.script
def IOU(boxes, anchors):
    # x, y, h, w
    anchors_mesh = torch.broadcast_to(anchors[:, :, None], (4, anchors.shape[1], boxes.shape[1]))
    boxes_mesh = torch.broadcast_to(boxes[:, None, :], (4, anchors.shape[1], boxes.shape[1]))
    # Calculate intersections and IoU
    w_i = torch.clip(.5 * (anchors_mesh[2] + boxes_mesh[2]) - torch.abs(boxes_mesh[0] - anchors_mesh[0]), min=torch.zeros_like(anchors_mesh[2]), max=torch.minimum(anchors_mesh[2],boxes_mesh[2]))
    h_i = torch.clip(.5 * (anchors_mesh[3] + boxes_mesh[3]) - torch.abs(boxes_mesh[1] - anchors_mesh[1]), min=torch.zeros_like(anchors_mesh[3]), max=torch.minimum(anchors_mesh[3],boxes_mesh[3]))
    I = w_i * h_i
    U = (boxes_mesh[2] * boxes_mesh[3]) + (anchors_mesh[2] * anchors_mesh[3]) - I
    IoU = I / U # n_anchors * n_boxes
    return IoU

# def label_anchors(image_info, feat_height, feat_width, base_anchors, feature_stride=cfg.FEATURE_STRIDE, A=cfg.A):
#     with torch.no_grad():
#         sp_anch = splashAnchors(feat_height, feat_width, 1, base_anchors, feature_stride, A=A)[0].T.to(cfg.DEVICE)
#         labels = torch.zeros(image_info.shape[0], sp_anch.shape[1], device=cfg.DEVICE)
#         values = torch.zeros(image_info.shape[0], 4, sp_anch.shape[1], device=cfg.DEVICE)
#         # 4 * n_anchors

#         for indx, boxes in enumerate(image_info):
#             if boxes.shape[0] > 0:
#                 boxes=boxes.T
#                 max_iou, max_indices = torch.max(IOU(boxes=boxes, anchors=sp_anch), dim=1) # Why yes, we do really need the indices

#                 # Classification object or not
#                 labels[indx] = torch.where(max_iou <= .05, -1, 0) + torch.where(max_iou >= .2, 1, 0)
#                 # -1 is negative, 0 is null and 1 is positive

#                 # Values for regressor
#                 boxes = boxes[:, max_indices]
#                 values[indx, 0] = (boxes[0] - sp_anch[0])/sp_anch[2]    # t_x
#                 values[indx, 1] = (boxes[1] - sp_anch[1])/sp_anch[3]    # t_y
#                 values[indx, 2] = torch.log(boxes[2]/sp_anch[2])        # t_w
#                 values[indx, 3] = torch.log(boxes[3]/sp_anch[3])        # t_h
#             else:
#                 labels[indx] = -torch.ones(sp_anch.shape[1])
        
#         return labels, values

def label_anchors(boxes, feat_height, feat_width, base_anchors, im_size, feature_stride=cfg.FEATURE_STRIDE, A=cfg.A, training = False):
    sp_anch = splashAnchors(feat_height, feat_width, 1, base_anchors, im_size, feature_stride, A=A, training=training)[0].T.to(cfg.DEVICE)
    labels = torch.zeros(sp_anch.shape[1], dtype=torch.float32, device=cfg.DEVICE)
    values = torch.zeros(4, sp_anch.shape[1], device=cfg.DEVICE)
    # 4 * n_anchors
    if boxes.shape[0] > 0:
        boxes=boxes.T
        max_iou, max_indices = torch.max(IOU(boxes=boxes, anchors=sp_anch), dim=1) # Why yes, we do really need the indices

        # Classification object or not
        labels = torch.where(max_iou <= .2, -1.0, 0.0) + torch.where(max_iou >= .6, 1.0, 0.0)
        # -1 is negative, 0 is null and 1 is positive

        # Values for regressor
        boxes = boxes[:, max_indices]
        values[0] = (boxes[0] - sp_anch[0])/sp_anch[2]    # t_x
        values[1] = (boxes[1] - sp_anch[1])/sp_anch[3]    # t_y
        values[2] = torch.log(boxes[2]/sp_anch[2])        # t_w
        values[3] = torch.log(boxes[3]/sp_anch[3])        # t_h
    else:
        labels = -torch.ones(sp_anch.shape[1])
    
    return labels, values




@torch.jit.script
def invert_values(values, anchors):
    
    values = values.permute(2,1,0)
    anchors = anchors.permute(2,1,0) 
    ret_vals = torch.empty_like(values)
    ret_vals[0] = anchors[2] * values[0] + anchors[0]
    ret_vals[1] = anchors[3] * values[1] + anchors[1]
    ret_vals[2] = anchors[2] * torch.exp(values[2])
    ret_vals[3] = anchors[3] * torch.exp(values[3])

    return ret_vals.permute(2,1,0)


# Fine guess I'll do it myself
def our_nms(anchors, scores, threshold=0.7):
    # Sort for funky stuff down the line
    _, sort_indexes = scores.sort(dim=1, descending=True)
    sort_indexes = sort_indexes.squeeze()
    anchors = anchors[sort_indexes.squeeze()].T

    # Get IoU, remove 1.0 from selfs and lower triangle
    IoUs = IOU(anchors, anchors)
    IoUs = IoUs - torch.eye(IoUs.shape[0], device=cfg.DEVICE)
    IoUsl = IoUs.tril()

    # Get maximum per row and then threshold (
    # In this way, since we are sorted by score,
    # we'll remove lower scoring boxes with index > 0.7,
    # this can be because of the lower triangle also)
    maximum, _ = IoUsl.max(dim=1)
    to_keep = (maximum <= threshold).nonzero()

    # Finally remove the bad stuff and return
    rows = IoUs.max(dim=1)[0][to_keep]
    return sort_indexes[to_keep[rows.sort(dim=0, descending=True)[1]].squeeze()]
    