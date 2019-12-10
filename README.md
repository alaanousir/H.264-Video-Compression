# H.264-Video-Compression

## To view and test the code

Open Report.ipynb and run the cells in it to view the report and part of the demo.
Open Demo.ipynb to view the whole pipeline executed.

## File description

*  Encoder.py has all the compression, and helper functions to video compression
*  Main.py gathers all encoder and decoder functions from Encoder.py and JPEG into 2 main functions: encode() and decode().
*  motion_estimation.jl performs motion estimation
*  binarize.jl converts the motion vectors, predicted_frames, and reference frame from numpy array to binary bit stream and vice versa.
*  bac.jl performs binary arithmetic encoding and decoding
*  bin_and_bac.jl gathers both binarize and BAC functions into 2 main functions: encode_bin_bac() and decode_bin_bac().