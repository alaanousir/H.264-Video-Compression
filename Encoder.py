import numpy as np 
import cv2
import sys
sys.path.insert(0,'JPEG-Compression')
import encoder as e

## The following functions will be called in the main.py of the video compression

# Reshape the image using the encoder functions in JPEG

# Second, get sub-images. Now, we have a frame that is divided into 16x16 macroblocks.

def motion_estimation(search_area, block):
    """
    Gets the search area from the reference frame, and the current 16x16 frame block
    and returns the motion vector by subtracting the matched MxN frame block in 
    the search area from the current MxN frame block.
    Args:
        search_area: a block of pixels from the reference frame
        block: a block of 16x16 pixels from the current frame. 
    Returns:
        motion_vector: the coordinate distance change (in pixel units) between the current frame and reference frame.
    """
    # find the matching 16x16 block from the search_area
    # matching can be obtained by finding the RMSE, or MAP.
    return motion_vector

def motion_estimation_to_all(prev_frame, current_frame):
    """
    Gets the reference frame (previous frame) and current frame. The prev_frame could be the k-nth frame where k is the current frame.
    and performs motion estimation on all 16x16 macroblocks in the current frame.
    Args:
        prev_frame: np array of the reference frame image divided into 16x16 macroblocks
        current_frame: current frame image divided into 16x16 macroblocks.
        - should have a shape of (X, macroblock_size, macroblock_size)
    returns:
        motion_vectors: np array of size(X, 2) where each macroblock has a motion vector represented in 2 values in the x and y coordinates 
    """
    #motion_vectors = np.array([motion_estimation(search_area, block)
    #                              for ......])
    return motion_vectors


def motion_prediction(prev_frame, motion_vectors):
    """ 
    Gets the previous frame and the motion_vectors and returns the predicted current frame.
    Args:
        prev_frame: np array of the reference frame image divided into 16x16 macroblocks 
        motion_vectors: np array of size(X, 2) that we got from motion_estimation block.
    Returns:
        predicted_frame: np array of the predicted frame
    """ 
    return predicted_frame

def residual(prev_frame, predicted_frame):
    """
    Gets the previous frame and the predicted frame from the motion prediction block
    and returns the residual
    Args:
        prev_frame: np array of the reference frame image divided into 16x16 macroblocks 
        predicted_frame: np array of the predicted frame
    Returns:
        residual_frame: np array of the residual frame.
    """
    return residual_frame
    
    
def spatial_model(residual_frame):
    """
    Gets the residual frame and applies DCT to it and returns the DCT coefficients.
    Args:
        residual_frame: np array of the residual frame of shape (X, macroblock_size, macroblock_size) that will be encoded
    returns:
        coefficients (numpy ndarray): 1d array representing the residual frame
    """
    return e.apply_dct_to_all(residual_frame)                     
    
    
def arithmetic_encode(coefficients, motion_vectors):
    """
    Applied arithmetic coding to the 1d array representing the bit stream
    Args:
        coefficients (numpy ndarray): 1d array representing the residual frame
        motion_vectors (numpy ndarray): 2d array represnting the motion vector (component in x and y direction) 
        
    Returns:
        acoded  (numpy ndarray): 1d array
    """