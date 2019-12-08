import Encoder as E
import numpy as np
from PIL import Image
import julia 
jl = julia.Julia()
jl.include('bac.jl')
jl.include('motion_estimation.jl')


def encode(path,predictedPerRef, no_frames = 1000,Resolution=1):
    vid_frame=E.get_video_frames(path,no_frames,Resolution)
    for i in range(len(vid_frame)):
        vid_frame[i]=E.interlace_comp_frames(vid_frame[i])
    ref_frames=vid_frame[::predictedPerRef]
    vid_mv=[]
    vid_residuals=[]
    for j in range(0,np.int(len(vid_frame)/predictedPerRef)):

        #Reshaping the reference frames to use in the coming blocks
        im_ref_y,_,_=E.get_sub_images(E.reshape_image(ref_frames[j][0]))
        im_ref_cb,_,_=E.get_sub_images(E.reshape_image(ref_frames[j][1]))
        im_ref_cr,_,_=E.get_sub_images(E.reshape_image(ref_frames[j][2]))

        for i in range(0,predictedPerRef-1):
            #Reshaping the current frame 
            current_im_blocks, nrows, ncols = E.get_sub_images(E.reshape_image(vid_frame[c][0]))
            
            #Motion estimation 
            mv = jl.motion_estimation(ref_frames[j][0], current_im_blocks, nrows, ncols)

            #Motion Compensation
            p_image_y = E.predict(im_ref_y,mv, ref_frames[0][0].shape[0], ref_frames[0][0].shape[1],16)

            p_image_cb=E.predict(im_ref_cb,mv, ref_frames[0][1].shape[0], ref_frames[0][1].shape[1],8)
            p_image_cr=E.predict(im_ref_cr,mv, ref_frames[0][1].shape[0], ref_frames[0][1].shape[1],8)

            #Calculating the residuals
            residual_frame=E.residual(vid_frame[c][0],p_image)
            
            #Spatial model 
            residual_blocks=E.spatial_model(residual_frame)

            

            # appending motion vectors and residual frames to change into bits
            vid_mv.append(mv)
            vid_residuals.append(residual_frame)
            
            c+=1
        c+=1
    #change into bitStream

    #encode using BAC
    Encoded_BitStream=jl.bitstream_bac_encode(Bitstream) 

    return Encoded_BitStream  

def decoder(Encoded_BitStream,predictedPerRef, no_frames = 1000,Resolution=1):
    Bitstream=jl.bitstream_bac_decode(Encoded_BitStream)
    
    #return residual, motion vectors, reference frame from bitstream

    #ref_frames has the same shape as the ref_frames variable in the encoder
    #mv has the same shape as the vid_mv variable in the encoder
    #vid_residuals has the same shape as vid_residuals in the encoder
    # all are need to be returned from the bitstream
    Reconstruced_frames=[]
    c=1
    for j in range(0,np.int(len(vid_frame)/predictedPerRef)):
        Reconstruced_frames.append(E.conv_decom_YUV2RGB(ref_frames[j]))

        #Reshaping the reference frames to use in the coming blocks
        im_ref,_,_=E.get_sub_images(E.reshape_image(ref_frames[j][0]))

        for i in range(0,predictedPerRef-1):
            #inverse spatial
            residual_blocks, n_rows, n_cols = E.spatial_inverse_model(vid_residuals[c])
            residual_frame = E.get_reconstructed_image(residual_blocks, n_rows, n_cols)

            #getting the predicted image
            p_image=E.predict(im_ref,vid_mv[c], ref_frames[0][0].shape[0], ref_frames[0][0].shape[1])

            #adding the residuals to get the reconstructed image
            Reconstructed=p_image+residual_frame
            c+=1

            

