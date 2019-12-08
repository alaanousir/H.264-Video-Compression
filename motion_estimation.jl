using Statistics # for mean function

rmse(x, y) = sqrt(mean((x .- y).^2))

function motion_estimation(ref_frame, current_block, block_num, n_rows, n_cols, search_size = 128)
     """
    Gets the search area from the reference frame, and the current 16x16 frame block
    and returns the motion vector by subtracting the matched MxN frame block in 
    the search area from the current MxN frame block.
    Args:
        ref_frame: the reference frame
        current_block: a block of 16x16 pixels from the current frame. 
        block_num: a tuple containing the i and j values of the image block. Corresponds to the row and column that the block
        resides in
        n_rows: number of rows in the frame
        n_cols: number of cols in the frame
        search_size: the size of the search_area as a whole.
    Returns:
        motion_vector: the coordinate distance change (in pixel units) between the current frame and reference frame.
    """
    # Pad the image with the search_size specified at the borders of the image
    # This ensures that the search area will never be outside the boundaries of the image
    h, w = size(ref_frame)
    border_width = div(search_size, 2) - 8
    
    # Initialize an array of zeros 
    padded = zeros(Int, h + border_width*2, w + border_width*2)
    # Add the reference frame to this array, so that we have a new frame with padded borders of size 64 on each edge
    # border_width + 1 because in julia, indexing starts from 1
    padded[border_width + 1 : h + border_width,
           border_width + 1 : w + border_width] = ref_frame
    
    # find the matching 16x16 block from the search_area
    # Convert this row and column number to x and y coordinates
    row, col = block_num
    block_size = size(current_block)[1]
    y, x = block_size*(row), block_size*(col)
    
     
    # the whole search area is 128*128
    # The coordinates of the padded np array is different from the coordinates of the current block.
    # i.e the top left pixel of the current block when mapped to padded np.array, it becomes (x+border_width,y+border_width)
    search_area = padded[y + 1 : y+ search_size, x + 1 : x + search_size]
    loss_prev = 1000
    y_moved::Int, x_moved::Int = -1, -1
    
    for j in 1:(search_size -block_size)
        for i in 1:(search_size - block_size)
            # Views uses up less time since it does not make a copy of search_area, and instead passes it by reference
            @views loss = rmse(current_block, search_area[j:j+block_size - 1,
                                                          i:i+block_size - 1])
            if loss < loss_prev
                loss_prev = loss
                y_moved, x_moved = (j,i)
            end
        end
    end
    
    motion_vectors = [ (y_moved-border_width -1) , (x_moved-border_width - 1) ]
    
    motion_vectors
end

function motion_estimation_to_all(ref_frame, current_frame, n_rows, n_cols, search_size =128)
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
    h, w = size(ref_frame)
    motion_vectors = zeros(Int, n_rows, n_cols,2)
    # Loop over the whole array of macroblocks
    # to get the motion vectors
    for col in 1:n_cols
        for row in 1:n_rows   
            @views motion_vectors[row, col, :] = motion_estimation(ref_frame, current_frame[(row - 1)*n_cols + col , :, :],
            (row - 1,col - 1), n_rows, n_cols, search_size)
        end
    end
    motion_vectors
end