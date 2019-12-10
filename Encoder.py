import numpy as np 
import cv2
import sys
sys.path.insert(0,'JPEG-Compression')
import encoder as e
import main as m
import decoder as d

## The following functions will be called in the main.py of the video compression

def get_video_frames(path, no_frames = 1000,Resolution=1):
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
            #Resize in case subpixel estimation is needed
            frameRGB=cv2.resize(frameRGB,(frameRGB.shape[1]*Resolution,frameRGB.shape[0]*Resolution))
            if ret == True:
                # Convert frame to YUV with 4:2:0 sampling
                frameYUV = cv2.cvtColor(frameRGB, cv2.COLOR_BGR2YUV_I420)

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
    image (np array): original image that needs to be reshaped 
    box_size (int): Size of the box sub images
    Returns:
    image_array (numpy ndarray, dtype = "uint8"): image reshaped to m x m
    np array.
    """
    n_rows = np.int(np.floor(image.shape[0]/box_size))
    n_cols = np.int(np.floor(image.shape[1]/box_size))

    image_array = cv2.resize(image, dsize=(n_cols*box_size, n_rows*box_size))
    return image_array

def interlace_comp_frames(complete_frames):
    """
    Gets: An array of complete frames and returns an array of interlaced complete frames. It takes the chroma components and 
        interlaces them together to prepare the frames for motion prediction based on a scaled version of the motion vectors.
    Args: Complete_frames: a list containing complete frames i.e. each element in a list of 5 image components. 
    Returns: an array of complete frames but with each containing 3 components, Y Cb (interlaced), Cr (interlaced), respectively.
    """
    c_rows, c_cols = complete_frames[0][1].shape * np.array([2,1])
    chroma_frames = []
    for frame in complete_frames:
        c_b = np.zeros((c_rows, c_cols), dtype= np.uint8)
        c_r = np.zeros((c_rows, c_cols), dtype= np.uint8)
        Cb1 = frame[1]
        Cb2 = frame[3]

        Cr1 = frame[2]
        Cr2 = frame[4]

        for r in range(c_rows):
            if r%2 == 0:
                c_b[r] = Cb1[np.int(r/2)]
                c_r[r] = Cr1[np.int(r/2)]
            else: 
                c_b[r] = Cb2[np.int(r/2)]
                c_r[r] = Cr2[np.int(r/2)]

        chroma_frames.append([frame[0],c_b,c_r])
        
    return chroma_frames

def deinterlace_comp_frames(interlaced_frames):
    
    c_rows, c_cols = interlaced_frames[1].shape * np.array([0.5,1])
    c_rows, c_cols = np.int(c_rows), np.int(c_cols)
    
    c_b = interlaced_frames[1]
    c_r = interlaced_frames[2]
    
    Cb1 = np.zeros((c_rows, c_cols), dtype= np.uint8)
    Cb2 = np.zeros((c_rows, c_cols), dtype= np.uint8)

    Cr1 = np.zeros((c_rows, c_cols), dtype= np.uint8)
    Cr2 = np.zeros((c_rows, c_cols), dtype= np.uint8)

    for r in range(c_rows*2):
        if r%2 == 0:
            Cb1[np.int(r/2)] = c_b[r]
            Cr1[np.int(r/2)] = c_r[r]
        else: 
            Cb2[np.int(r/2)] = c_b[r]
            Cr2[np.int(r/2)] = c_r[r]
            
    
        
    return [interlaced_frames[0],Cb1,Cr1,Cb2,Cr2]

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


def predict(image_blocks, motion_vecs, p_rows, p_cols, block_size = 16):
    """
    Gets: An array of serial image blocks with each block of size block_size, block_size and constructs an image of each block moved by
    a corresponding motion vector.
    Args: 
        image_blocks: 1D array of image block_size x block_size blocks 
        motion_vecs: motion vectors corresponding to the blocks in image_blocks.
        p_rows: rows of predicted frame (constant for all frames)
        p_cols: columns of predicted frame (constant for all frames)
    Returns: 
        predicted_image: an image where each block has been moved to its predicted place according to its motion vector
    """
    predicted_image = d.get_reconstructed_image(image_blocks, np.int(p_rows/block_size), np.int(p_cols/block_size), box_size=block_size)
    image_blocks = image_blocks.reshape(np.int(p_rows/block_size),np.int(p_cols/block_size),block_size,block_size)   #contruct the image first with no movements
    
    for i in range(np.int(p_rows/block_size)):
        for j in range(np.int(p_cols/block_size)):
            vector = motion_vecs[i,j]
            # checking for image boundaries to avoid any out of bound indecies 
            if i*block_size + vector[1] + block_size <= p_rows and i*block_size + vector[1] >=0 and j*block_size + vector[0] + block_size <= p_cols and j*block_size + vector[0] >= 0:
                # move only the blocks where motion vector is not 0 
                if vector[0] != 0 or vector[1] != 0:
                    predicted_image[i*block_size + vector[1] : i*block_size + vector[1] + block_size, j*block_size + vector[0] : j*block_size + vector[0] + block_size] = image_blocks[i,j]
                    
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
    return current_frame.astype(int) - predicted_frame.astype(int)
            
    
    
def spatial_model(residual_frame, box_size):
    """
    Gets the residual frame, converts it into 8x8 or 16x16 blocks and applies DCT to it and returns the DCT coefficients.
    Args:
        residual_frame: np array of the residual frame of shape (X, macroblock_size, macroblock_size) that will be encoded
        box_size: size of blocks.
    Returns:
        quantized_coeff (numpy ndarray): 1d array representing the residual frame
    """
    residual_blocks, n_rows, n_cols = e.get_sub_images(residual_frame,box_size)
    coeff = e.apply_dct_to_all(residual_blocks)
    if box_size == 16:
        table =  m.table_16_low
    else:
        table = m.table_8_low        
    quantized_coeff = e.quantize(coeff, table)
    return e.run_length_code(e.serialize(quantized_coeff))

def spatial_inverse_model(quantized_coeff, n_rows, n_cols, box_size):
    """
    Gets the quantized coefficients and returns the reconstructed residual frame
    Args:
        quantized_coeff: np array of the quantized coefficients of shape (X, block_size, block_size) that will be encoded
        n_rows: number of rows
        n_cols: number of columns
    Returns:
        reconstructed_residual (numpy ndarray): 1d array representing the residual frame
    """
    if box_size == 16:
        table =  m.table_16_low
    else:
        table = m.table_8_low 
    quantized_coeff = d.run_length_decode(quantized_coeff)
    quantized_coeff = d.deserialize(quantized_coeff, n_rows*n_cols, box_size, box_size)
    dequantized_coeff=d.dequantize(quantized_coeff, table)
    divided_image = d.apply_idct_to_all(dequantized_coeff)
    return d.get_reconstructed_image(divided_image, n_rows, n_cols, box_size)



def conv_decom_YUV2RGB(complete_frame):
    """
    Gets a list containing all the components of a YUV in this order [Y, Cb1, Cr1, Cb2, Cr2], then combine them all together
    and convert them to their RGB equivilant
    Args:
        complete_frame: a list of the 5 YUV components
    Returns:
        an RGB OpenCV frame (3d numpy array) that is ready to be shown with cv2.imshow()
    """
    rows, cols = complete_frame[0].shape[0]+complete_frame[1].shape[0]*2, complete_frame[0].shape[1]
    Y_row = np.int(rows - rows*1/3)
    
    frame1 = np.zeros((rows,cols), dtype= np.uint8)
    frame1[0:Y_row, : ] = complete_frame[0]
    frame1[Y_row:np.int(Y_row*1.25),0: np.int(cols/2)] = complete_frame[1]
    frame1[np.int(Y_row*1.25):np.int(Y_row*1.5), 0: np.int(cols/2)] = complete_frame[2]
    frame1[Y_row:np.int(Y_row*1.25), np.int(cols/2):]  = complete_frame[3]
    frame1[np.int(Y_row*1.25):np.int(Y_row*1.5), np.int(cols/2):] = complete_frame[4]
    
    return cv2.cvtColor(frame1, cv2.COLOR_YUV2BGR_I420)

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
