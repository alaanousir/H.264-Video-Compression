import numpy as np 
import cv2

def motion_estimation(search_area, block):
    """
    Gets the search area from the reference frame, and the current MxN frame block
    and returns the residual block by subtracting the matched MxN frame block in 
    the search area from the current MxN frame block.
    Args:
        search_area: a block of pixels from the reference frame
        block: a block of MxN pixels from the current frame. 
    Returns:
        residual: the matched MxN pixels in the previous reference frame
    """
    # First, find the matching MxN block from the search_area
    # Then subtract the matched reference block from the current block

def motion_estimation_to_all(ref_frame, frame):
    """
    Gets the reference frame (previous frame) and current frame
    and performs motion estimation on all MxN blocks in the current frame.
    Args:
        ref_frame: np array of the reference frame image
        frame: current frame image
    returns:
        residual_frame: the residual frame that will be encoded
    """
    #residual_frame = np.array([motion_estimation(search_area, block)
    #                              for ......])