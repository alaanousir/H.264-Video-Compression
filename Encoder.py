import numpy as np 
import cv2
import sys
sys.path.insert(0,'JPEG-Compression')
import encoder as e

## The following functions will be called in the main.py of the video compression

# Reshape the image using the encoder functions in JPEG

# Second, get sub-images. Now, we have a frame that is divided into 16x16 macroblocks.


def rmse(im, ref_im):
    """
    Gets the root mean squared error between 2 images
    Args:
         im (numpy array) : The current image
         ref_im (numpy ndarray): The reference image
    Returns:
        rmse: root mean squared error as a metric to compare between the original image and the reconstructed
    """
    error = im - reconstructed_image
    mse = np.sum(np.square(error)) / (im.shape[0] * im.shape[1])
    rmse = np.sqrt(mse)

    return rmse

def motion_vectors(coordinates, matching_coordinates , block_size):
    """
    Gets the coordinates of the current image block, and the coordinates of the matching search area
    and returns the motion vector
    Args:
       coordinates: a tuple containing the x and y coordinates of the image block
       matching_coordinates: a tuple containing the i and j coordinates of the matching image block
    Returns:
        motion_vector: the x and y distances between the x-y coordinates of the image block and the i-j coordinates of the matching block
    """
    if x>i:
        x_dir = x-i+0.5*block_size
    else:
        x_dir = i-x+0.5*block_size
    if y>j:
        y_dir = y-j+0.5*block_size
    else:
        y_dir = j-y+0.5*block_size
        
    return (y_dir,x_dir)

def motion_estimation(ref_frame, current_block, coordinates, search_size =64):
    """
    Gets the search area from the reference frame, and the current 16x16 frame block
    and returns the motion vector by subtracting the matched MxN frame block in 
    the search area from the current MxN frame block.
    Args:
        ref_frame: the reference frame
        block: a block of 16x16 pixels from the current frame. 
        coordinates: a tuple containing the x and y coordinates of the image block
        search_size: is half the size of the search_area as a whole.
    Returns:
        residual: np array of the residual macroblock
        motion_vector: the coordinate distance change (in pixel units) between the current frame and reference frame.
    """
    ###############
    #SPECIAL CASES NOT HANDLED YET
    ############
               
    # find the matching 16x16 block from the search_area
    y, x = coordinates
    search_area = ref_frame[(y - search_size) : (y + search_size), (x - search_size) : (x + search_size)]
    block_size = current_block.shape[0]
    loss_prev = 1000
    for j in range(search_size - block_size):
        for i in range(search_size - block_size):
            loss = rmse(current_block, search_area[i:i+block_size,i:i+block_size])
            if loss < loss_prev:
                loss_prev = loss
                matching_coordinates = (j,i)
    motion_vectors = motion_vectors(coordinates , matching_coordinates , block_size)     
    
    return residual, vector_vectors
   

def motion_estimation_to_all(prev_frame, current_frame):
    """
    Gets the reference frame (previous frame) and current frame. The prev_frame could be the k-nth frame where k is the current frame.
    and performs motion estimation on all 16x16 macroblocks in the current frame.
    Args:
        prev_frame: np array of the reference frame image divided into 16x16 macroblocks
        current_frame: current frame image divided into 16x16 macroblocks.
        - should have a shape of (X, macroblock_size, macroblock_size)
    returns:
        residuals: np array of the residual macroblocks.
        motion_vectors: np array of size(X, 2) where each macroblock has a motion vector represented in 2 values in the x and y coordinates 
    """
    #motion_vectors, residuals = np.array([motion_estimation(search_area, block)
    #                              for ......])
    return residuals, motion_vectors


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