import torch
import numpy as np


def generate_anchors(base, ratios, scale) : 

    '''

    A = len(ratios) * len(scales)

    args : 
        base   : int : W and H of windows
        ratios : array : ratios to apply to base
        scales : array : scales to apply to base
    return : 
        anchors ndarray : (A,4): {x_min, y_min, x_max, y_max}
    '''    
    pass



def splashAnchors(anchors): 
    pass