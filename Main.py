import numpy as np
from PIL import Image
import encoder as e
import Encoder as E

def encode(path,predictedPerRef, no_frames = 1000):
    vid_frame=E.get_video_frames(path,no_frames)
    ref_frames=vid_frame[::predictedPerRef]
    vid_mv=[]
    vid_residuals=[]
    for j in range(0,np.int(len(vid_frame)/predictedPerRef)):

        #Reshaping the reference frames to use in the coming blocks
        im_ref,_,_=E.get_sub_images(E.reshape_image(ref_frames[j][0]))

        for i in range(0,predictedPerRef-1):
            #Reshaping the current frame 
            current_im_blocks, nrows, ncols = E.get_sub_images(E.reshape_image(vid_frame[c][0]))
            
            #Motion estimation 
            mv = E.motion_estimation_to_all(ref_frames[j][0], current_im_blocks, nrows, ncols)

            #Motion Compensation
            p_image = E.predict(im_ref,block_mv, ref_frames[0][0].shape[0], ref_frames[0][0].shape[1])

            #Calculating the residuals
            residual_frame=E.residual(vid_frame[c][0],p_image)

            # appending motion vectors and residual frames to change into bits
            vid_mv.append(mv)
            vid_residuals.append(residual_frame)
            
            c+=1
        c+=1
       


        

