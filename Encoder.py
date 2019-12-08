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
     """
    Gets an image of arbitrary size
    and returns a reshaped array of (box_size, box_size) elements
    Args:
        image (np arrat): original image that needs to be reshaped 
        box_size (int): Size of the box sub images
    Returns:
        image_array (numpy ndarray, dtype = "uint8"): image reshaped to m x m
        np array.
    """
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
    Returns:
        serialized_coeff (numpy ndarray): 1d array representing the residual frame
    """
    
    coeff = e.apply_dct_to_all(residual_frame)
    quantized_coeff = e.quantize(coeff, m.table_8_high)
    serialized_coeff=e.serialize(quantized_coeff)
    return serialized_coeff

def spatial_inverse_model(serialized_coeff):
    """
    Gets the serialized coefficients and deserialize it and applies IDCT.
    Args:
        serialized_coeff (numpy ndarray): 1d array representing the residual frame
    Returns:
        frame blocks (np array)
    """
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
