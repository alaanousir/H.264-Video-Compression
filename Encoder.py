import numpy as np 
import cv2
import sys
sys.path.insert(0,'JPEG-Compression')
import encoder as e
import main as m
import decoder as d

## The following functions will be called in the main.py of the video compression

def get_video_frames(path, no_frames = 1000):
    """
    Gets a path to the video to be read
    Args:
        path: string to the path of the video
        no_frames: int, specifies the number of frames to be read from the video
    Returns:
        a list of complete frames. Each complete frame is a list containing the Y,Cb,Cr components of each frame
    """
    vid = cv2.VideoCapture(path)
    # Initialize a np array to hold all frames.
    vid_frame = []
    # Read until video is completed
    for i in range(no_frames):
            if vid.isOpened() == 0:
                print("couldn't open video")
            # Capture frame-by-frame
            ret, frameRGB = vid.read()
            if ret == True:
                # Convert frame to YUV with 4:2:0 sampling
                frameYUV = cv2.cvtColor(frameRGB, cv2.COLOR_RGB2YUV_I420)

                # Get frame components
                rows, cols = frameYUV.shape 
                Y_row = np.int(rows - rows*1/3)
                frame_Y = frameYUV[0:Y_row, :]

                frame_Cb1 = frameYUV[Y_row:np.int(Y_row*1.25),0: np.int(cols/2)]
                frame_Cr1 = frameYUV[np.int(Y_row*1.25):np.int(Y_row*1.5), 0: np.int(cols/2)]

                frame_Cb2 = frameYUV[Y_row:np.int(Y_row*1.25), np.int(cols/2):]
                frame_Cr2 = frameYUV[np.int(Y_row*1.25):np.int(Y_row*1.5), np.int(cols/2):]

                complete_frame = np.array([frame_Y,frame_Cb1,frame_Cr1,frame_Cb2,frame_Cr2])

                # Add frame to list of frames
                vid_frame.append(complete_frame)
            # Break the loop
            else: 
                break
    return vid_frame

def reshape_image(image, box_size = 16):

    n_rows = np.int(np.floor(image.shape[0]/box_size))
    n_cols = np.int(np.floor(image.shape[1]/box_size))

    image_array = cv2.resize(image, dsize=(n_cols*box_size, n_rows*box_size))
    return image_array

def get_sub_images(image_array, box_size=16):
    """
    Gets a grayscale image and returns an array of (box_size, box_size) elements
    Args:
        image_array (numpy ndarray): Image input we want to divide to box
                                     sub_images.
         Should have shape (length, width, n_channels) where length = width
          e. g. n_channels = 3 for RGB
         box_size (int): Size of the box sub images
    Returns:
        divided_image (numpy ndarray, dtype = "uint8"): array of divided images
         - should have a shape of (X, box_size, box_size, n_channels).
        n_rows: number of rows or blocks
        n_cols: number of columns in image
          the number of blocks is n_rows*n_cols
    """
    n_rows = np.int(image_array.shape[0]/box_size)
    n_cols = np.int(image_array.shape[1]/box_size)

    # make the image into a square to simplify operations based
    #  on the smaller dimension
    # d = min(n_cols, n_rows)

    # Note: images are converted to uint8 datatypes since they range between
    #  0-255. different datatypes might misbehave (based on my trials)
    image_blocks = np.asarray([np.zeros((box_size, box_size), dtype='uint8')
                               for i in range(n_rows*n_cols)], dtype='uint8')

    # break down the image into blocks
    c = 0
    for i in range(n_rows):
        for j in range(n_cols):
            image_blocks[c] = image_array[i*box_size: i*box_size+box_size,
                                          j*box_size:j*box_size+box_size]
            c += 1

    # If you want to reconvert the output of this function into images,
    #  use the following line:
    # block_image = Image.fromarray(output[idx])

    return image_blocks, n_rows, n_cols
    

def rmse(im, ref_im):
    """
    Gets the root mean squared error between 2 images
    Args:
         im (numpy array) : The current image
         ref_im (numpy ndarray): The reference image
    Returns:
        rmse: root mean squared error as a metric to compare between the original image and the reconstructed
    """
    error = ref_im - im
    mse = np.sum(np.square(error)) / (im.shape[0] * im.shape[1])
    rmse = np.sqrt(mse)

    return rmse


def motion_estimation(ref_frame, current_block, block_num, n_rows, n_cols, search_size = 128):
    """
    Gets the search area from the reference frame, and the current 16x16 frame block
    and returns the motion vector by subtracting the matched MxN frame block in 
    the search area from the current MxN frame block.
    Args:
        ref_frame: the reference frame
        current_block: a block of 16x16 pixels from the current frame. 
        block_num: a tuple containing the i and j values of the image block. Corresponds to the row and column that the block
        resides in
        search_size: the size of the search_area as a whole.
    Returns:
        motion_vector: the coordinate distance change (in pixel units) between the current frame and reference frame.
    """
    # Pad the image with the search_size specified at the borders of the image
    # This ensures that the search area will never be outside the boundaries of the image
    h, w = ref_frame.shape
    border_width = int(search_size/2) - 8
    padded = np.zeros(( h + border_width*2, w + border_width*2))  # Initialize an array of zeros 
    # Add the reference frame to this array, so that we have a new frame with padded borders of size 64 on each edge
    padded[border_width : h+border_width, border_width: w+border_width] = ref_frame 
    
    # find the matching 16x16 block from the search_area
    # Convert this row and column number to x and y coordinates
    row, col = block_num
    block_size = current_block.shape[0]
    y, x = (block_size*(row), block_size*(col))
    
    # the whole search area is 128*128
    # The coordinates of the padded np array is different from the coordinates of the current block.
    # i.e the top left pixel of the current block when mapped to padded np.array, it becomes (x+border_width,y+border_width)
    search_area = padded[y : (y+ search_size), x : (x + search_size)]
    
    loss_prev = 1000
    c = 0
    for j in range(search_size -block_size):
        for i in range(search_size - block_size):
            c+=1
            loss = rmse(current_block, search_area[j:j+block_size,i:i+block_size])
            if loss < loss_prev:
                loss_prev = loss
                y_moved, x_moved = (j,i)
    
    motion_vectors = (y_moved-border_width, x_moved-border_width)   
    
    return motion_vectors
   

def motion_estimation_to_all(ref_frame, current_frame, n_rows, n_cols, search_size =128):
    """
    Gets the reference frame (previous frame) and current frame. The prev_frame could be the k-nth frame where k is the current frame.
    and performs motion estimation on all 16x16 macroblocks in the current frame.
    Args:
        ref_frame: np array of the reference frame image
        current_frame: current frame image divided into 16x16 macroblocks of shape 
        - should have a shape of (X, macroblock_size, macroblock_size)
        n_rows: number of rows
        n_cols: number of columns
        search_size: the search area is search_size*2
    returns:
        motion_vectors: np array of size(X, 2) where each macroblock has a motion vector represented in 2 values in the x and y coordinates 
    """
    h, w = ref_frame.shape
    macroblock_size = current_frame.shape[1]
    motion_vectors = []
    # Loop over the whole array of macroblocks
    # to get the motion vectors
    for row in range(n_rows):
        for col in range(n_cols):
            motion_vectors.append(motion_estimation(ref_frame, current_frame[row*n_cols + col], (row,col), n_rows, n_cols, search_size))
                                  
    return motion_vectors



def predict(image_blocks, motion_vecs, p_rows, p_cols):
    """
    Gets: An array of serial image blocks with each block of size 16, 16 and constructs an image of each block moved by
    a corresponding motion vector.
    Args: 
        image_blocks: 1D array of image 16x16 blocks 
        motion_vecs: motion vectors corresponding to the blocks in image_blocks.
        p_rows: rows of predicted frame (constant for all frames)
        p_cols: columns of predicted frame (constant for all frames)
    Returns: 
        predicted_image: an image where each block has been moved to its predicted place according to its motion vector
    """
    predicted_image = get_reconstructed_image(image_blocks, np.int(p_rows/16), np.int(p_cols/16), box_size=16)
    image_blocks = image_blocks.reshape(np.int(p_rows/16),np.int(p_cols/16),16,16)   #contruct the image first with no movements
    
    for i in range(np.int(p_rows/16)):
        for j in range(np.int(p_cols/16)):
            vector = motion_vecs[i,j]
            # checking for image boundaries to avoid any out of bound indecies 
            if i*16 + vector[1] + 16 <= p_rows and i*16 + vector[1] >=0 and j*16 + vector[0] + 16 <= p_cols and j*16 + vector[0] >= 0:
                # move only the blocks where motion vector is not 0 
                if vector[0] != 0 or vector[1] != 0:
                    predicted_image[i*16 + vector[1] : i*16 + vector[1] + 16, j*16 + vector[0] : j*16 + vector[0] + 16] = image_blocks[i,j]
                    
    return predicted_image



def residual(current_frame, predicted_frame):
    """
    Gets the current frame and predicted_frame and subtracts them to return the residual frame.
    Args: current_frame: current frame image divided into 16x16 macroblocks.
        - should have a shape of (X, macroblock_size, macroblock_size)
          predicted_frame: np array of the predicted frame
    Returns: 
          residual_frame: np array of the residual macroblock
    """      
    return current_frame - predicted_frame 
            
    
    
def spatial_model(residual_frame):
    """
    Gets the residual frame and applies DCT to it and returns the DCT coefficients.
    Args:
        residual_frame: np array of the residual frame of shape (X, macroblock_size, macroblock_size) that will be encoded
    returns:
        quantized_coeff (numpy ndarray): 1d array representing the residual frame
    """
    
    coeff = e.apply_dct_to_all(residual_frame)
    quantized_coeff = e.quantize(coeff, m.table_8_high)
    serialized_coeff=e.serialize(quantized_coeff)
    return serialized_coeff

def spatial_inverse_model(serialized_coeff):
    quantized_coeff=d.deserialize(serialized_coeff)
    dequantized_coeff=d.dequantize(quantized_coeff)
    return d.apply_idct_to_all(dequantized_coeff)

def get_reconstructed_image(divided_image, n_rows, n_cols, box_size=8):
    """
    Gets an array of (box_size,box_size) pixels
    and returns the reconstructed image
    Args:
        divided_image (numpy ndarray, dtype = "uint8"): array of divided images
        n_rows: number of rows or blocks
        n_cols: number of columns in image
            the number of blocks is n_rows*n_cols
        box_size (int): Size of the box sub images
    Returns:
        reconstructed_image (numpy ndarray): Image reconstructed from the array
        of divided images.

    """
    image_reconstructed = np.zeros((n_rows*box_size, n_cols*box_size), dtype=np.uint8)
    c = 0
    # break down the image into blocks
    for i in range(n_rows):
        for j in range(n_cols):
            image_reconstructed[i*box_size: i*box_size+box_size,
                                j*box_size: j*box_size+box_size] = divided_image[c]
            c += 1
            
    # If you want to reconvert the output of this function into images,
    #  use the following line:
    # block_image = Image.fromarray(output[idx])

    return image_reconstructed

def reconstructed(predicted_frame, quantized_coeff):
    """
    Gets the predicted_frame and quantized_coefficients and transforms back the coefficients to residual frame 
    and gets the reconstructed image by adding the predicted_frame to the reconstructed residual frame.
    Args:
        predicted_frame: np array of the predicted frame
        quantized_coeff (numpy ndarray): 1d array representing the residual frame
    Returns:
        reconstructed_current
    """
    #return reconstructed_current
    
    
def arithmetic_encode(quantized_coeff, motion_vectors):
    """
    Applied arithmetic coding to the 1d array representing the bit stream
    Args:
        quantized_coeff (numpy ndarray): 1d array representing the residual frame
        motion_vectors (numpy ndarray): 2d array represnting the motion vector (component in x and y direction) 
        
    Returns:
        acoded  (numpy ndarray): 1d array
    """