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
    #test
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
        residual_frame: np array of the residual frame that will be encoded
    """
    #residual_frame = np.array([motion_estimation(search_area, block)
    #                              for ......])

def spatial_model(residual_frame):
    """
    Gets the residual frame and applies DCT to it and returns the DCT coefficients.
    Args:
        residual_frame: np array of the residual frame that will be encoded
    returns:
        coefficients (numpy ndarray): 1d array representing the residual frame
    """
    
def arithmetic_encode(coefficients, vectors):
    """
    Applied arithmetic coding to the 1d array representing the bit stream
    Args:
        coefficients (numpy ndarray): 1d array representing the residual frame
        vectors (numpy ndarray): 2d array represnting the motion vector (component in x and y direction) 
        
    Returns:
        acoded  (numpy ndarray): 1d array
    """