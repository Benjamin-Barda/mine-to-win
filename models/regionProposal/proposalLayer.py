import numpy as np


class _proposal:

    def __init__(self):
        pass

    def __call__(self, fg_scores, reg_scores, anchors, img_size):
        '''
        args : 
            fg_score : ()
            reg_scores
            anchors
            img_size
        return : 
            RoI
        
        Alg : 
            Convert Anchors into proposal ... that is apply offsets from reg to anchor and clip it to image
            Sort them based on fg_scores
            Apply NMS and take only top K
        '''

        # Apply predicted offset to original anchors thus turning them into proposals
        rois = getROI(anchors, reg_scores)





def getROI(src_box, offset):
    """
    Taken from R-CNN paper

    offset calc  :
        dst_x = src_w * off_x + src_x
        dst_y = src_h * off_y + src_y
        dst_w = src_w * exp(off_w)
        dst_h = src_h * exp(off_h)

    """

    src_x = src_box[:, 0]
    src_y = src_box[:, 1]
    src_w = src_box[:, 2]
    src_h = src_box[:, 2]

    off_x = offset[:, 0]
    off_y = offset[:, 1]
    off_w = offset[:, 2]
    off_h = offset[:, 3]

    dst_x = src_w * off_x + src_x
    dst_y = src_h * off_y + src_y
    dst_w = src_w * np.exp(off_w)
    dst_h = src_h * np.exp(off_h)

    dst = np.vstack((dst_x, dst_y, dst_w, dst_h)).T

    return dst
