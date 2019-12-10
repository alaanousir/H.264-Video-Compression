import Encoder as E
import numpy as np
from PIL import Image
import julia 
import math
jl = julia.Julia()
jl.include('bac.jl')
jl.include('motion_estimation.jl')


def encode(path, predictedPerRef, no_frames = 5,Resolution=1):
    """
    Gets the path to the video to be encoded, and returns an encoded video in bits
    Args:
        path (string): string containing the path to the image
        predictedPerRef (int): integer value that specifies the number of P (predicted) frames for each I (reference) frame
        Resolution (int): integer value that specifies the depth of the subpixel estimation
    Returns:
        Encoded_BitStream: binary arithmetic encoded motion vectors, predicted frames, and reference frame.
    """
    vid_frame=E.get_video_frames(path,no_frames,Resolution)
    vid_frame=E.interlace_comp_frames(vid_frame)
    ref_frames=vid_frame[::predictedPerRef]
    n_predicted_frames = no_frames - int(np.ceil(len(vid_frame)/(predictedPerRef+1)))
    
    #pre-allocate residuals and motion vectors
    residual_frames_y = np.zeros((n_predicted_frames,
    vid_frame[0][0].shape[0], vid_frame[0][0].shape[1]), dtype = int)

    residual_frames_cb = np.zeros((n_predicted_frames,
    vid_frame[0][1].shape[0], vid_frame[0][1].shape[1]), dtype = int)

    residual_frames_cr= np.zeros((n_predicted_frames,
    vid_frame[0][2].shape[0], vid_frame[0][2].shape[1]), dtype = int)

    vid_mv = np.zeros((n_predicted_frames, int(vid_frame[0][0].shape[0]/16),
     int(vid_frame[0][0].shape[1]/16), 2), dtype = int)

    c=1
    for j in range(0,math.ceil(len(vid_frame)/predictedPerRef)):

        #Reshaping the reference frames to use in the coming blocks
        im_ref_y,_,_=E.get_sub_images(E.reshape_image(ref_frames[j][0]))
        im_ref_cb,_,_=E.get_sub_images(E.reshape_image(ref_frames[j][1],8),8)
        im_ref_cr,_,_=E.get_sub_images(E.reshape_image(ref_frames[j][2],8),8)
        

        for i in range(0,predictedPerRef-1):
            #Reshaping the current frame 
            current_im_blocks, nrows, ncols = E.get_sub_images(E.reshape_image(vid_frame[c][0]))
            
            #Motion estimation 
            mv = jl.motion_estimation_to_all(ref_frames[j][0], current_im_blocks, nrows, ncols)

            #Motion Compensation
            p_image_y = E.predict(im_ref_y,mv, ref_frames[0][0].shape[0], ref_frames[0][0].shape[1],16)
            mv_cb=np.zeros(mv.shape,dtype=int)
            for x in range(mv.shape[0]):
                for z in range(mv.shape[1]):
                    mv_cb[x][z][0]=np.int(mv[x][z][0]/2)
                    mv_cb[x][z][1]=np.int(mv[x][z][1]/2)

            p_image_cb=E.predict(im_ref_cb,mv_cb, ref_frames[0][1].shape[0], ref_frames[0][1].shape[1],8)
            p_image_cr=E.predict(im_ref_cr,mv_cb, ref_frames[0][2].shape[0], ref_frames[0][2].shape[1],8)

            #Calculating the residuals
            res_index = int(c - np.ceil(j/predictedPerRef) - 1)
            residual_frames_y[res_index] = E.residual(vid_frame[c][0],p_image_y)
            residual_frames_cb[res_index] = E.residual(vid_frame[c][1],p_image_cb)
            residual_frames_cr[res_index] = E.residual(vid_frame[c][2],p_image_cr)
            
            vid_mv[res_index] = mv
            c+=1
            if(c>len(vid_frame)-1):
                break
        c+=1
    # Delete unwanted variables 
    del p_image_cb, p_image_cr
    
    # Perform the spatial model
    quantized_coeff_y = []
    quantized_coeff_cb = []
    quantized_coeff_cr = []
    for j in range(0, n_predicted_frames):
        quantized_coeff_y.append(E.spatial_model(residual_frames_y[j], 16))
        
        quantized_coeff_cb.append(E.spatial_model(residual_frames_cb[j], 8))
        
        quantized_coeff_cr.append(E.spatial_model(residual_frames_cr[j], 8))
    #change into bitStream
    #encode using BAC
    Encoded_BitStream= jl.encode_bin_bac(vid_mv,predictedPerRef, no_frames, (quantized_coeff_y,quantized_coeff_cb,quantized_coeff_cr), ref_frames) 

    return Encoded_BitStream  

def decode(Encoded_BitStream,predictedPerRef, no_frames = 5,Resolution=1):
    """
    Gets the encoded bitstream and returns the reconstructed frames.
    Args:
        Encoded_BitStream: binary arithmetic encoded motion vectors, predicted frames, and reference frame.
        predictedPerRef (int): integer value that specifies the number of P (predicted) frames for each I (reference) frame
        no_frames: number of incoming frames.
        Resolution (int): integer value that specifies the depth of the subpixel estimation
    Returns:
        ref_frames: np array of the reference frames
        Reconstructed_frames: np array of the reconstructed frames
    """
   
    # Decode and convert bitstream to motion vectors, residuals, and ref_frame
    vid_mv, quantized_coeff, ref_frames = decode_bin_bac(Encoded_BitStream)
    
    quantized_coeff_y,quantized_coeff_cb,quantized_coeff_cr = quantized_coeff
    del quantized_coeff
    
    n_predicted_frames = no_frames - int(np.ceil(no_frames/predictedPerRef))
    nrows = np.int(ref_frames[0][0].shape[0]/16)
    ncols = np.int(ref_frames[0][0].shape[1]/16)
    
    height = ref_frames[0][0].shape[0]
    width = ref_frames[0][0].shape[1]
    
    # Inverse the spatial model
    quantized_residual_y = np.zeros((n_predicted_frames, height, width), dtype = int)
    quantized_residual_cb = np.zeros((n_predicted_frames, np.int(height/2), np.int(width/2)), dtype = int)
    quantized_residual_cr = np.zeros((n_predicted_frames, np.int(height/2), np.int(width/2)), dtype = int)
    
    for j in range(0,predictedPerRef-1):
        quantized_residual_y[j] = E.spatial_inverse_model(quantized_coeff_y[j], nrows, ncols, 16)

        quantized_residual_cb[j] =  E.spatial_inverse_model(quantized_coeff_cb[j], nrows, ncols, 8)

        quantized_residual_cr[j] = E.spatial_inverse_model(quantized_coeff_cr[j], nrows, ncols, 8) 

        
    del quantized_coeff_y,quantized_coeff_cb,quantized_coeff_cr 

    Reconstruced_frames=[]
    c=0
    for j in range(0,np.int(no_frames/predictedPerRef)):
        Reconstruced_frames.append(E.conv_decom_YUV2RGB(E.deinterlace_comp_frames(ref_frames[j])))

        #Reshaping the reference frames to use in the coming blocks
        im_ref_y,_,_=E.get_sub_images(E.reshape_image(ref_frames[j][0]))
        im_ref_cb,_,_=E.get_sub_images(E.reshape_image(ref_frames[j][1],8),8)
        im_ref_cr,_,_=E.get_sub_images(E.reshape_image(ref_frames[j][2],8),8)
        for i in range(0,predictedPerRef-1):
            #inverse spatial
            #residual_blocks, n_rows, n_cols = E.spatial_inverse_model(vid_residuals[c])
            #residual_frame = E.get_reconstructed_image(residual_blocks, n_rows, n_cols)
            mv=vid_mv[c]
            for i in range(mv.shape[0]):
                for j in range(mv.shape[1]):
                    mv_cb[i][j][0]=np.int(mv[i][j][0]/2)
                    mv_cb[i][j][1]=np.int(mv[i][j][1]/2)
            #getting the predicted image
            p_image_y = E.predict(im_ref_y, mv, ref_frames[0][0].shape[0], ref_frames[0][0].shape[1])
            p_image_cb = E.predict(im_ref_cb, mv_cb, ref_frames[0][1].shape[0], ref_frames[0][1].shape[1], 8)
            p_image_cr = E.predict(im_ref_cr, mv_cb, ref_frames[0][2].shape[0], ref_frames[0][2].shape[1], 8)
            #adding the residuals to get the reconstructed image

            Reconstructed_y = p_image_y + quantized_residual_y[c]
            Reconstructed_cb = p_image_cb + quantized_residual_cb[c]
            Reconstructed_cr = p_image_cr + quantized_residual_cr[c]

            Reconstructed_interlaced=[Reconstructed_y,Reconstructed_cb,Reconstructed_cr]
            Reconstruced_frames.append(E.conv_decom_YUV2RGB(E.deinterlace_comp_frames(Reconstructed_interlaced)))
            c+=1
    return ref_frames, Reconstructed_frames


