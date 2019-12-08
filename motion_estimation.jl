using Statistics # for mean function
rmse(x, y) = sqrt(mean((x .- y).^2))

function motion_estimation(ref_frame, current_block, block_num, n_rows, n_cols, search_size = 128)
    h, w = size(ref_frame)
    border_width = div(search_size, 2) - 8
    padded = zeros(Int, h + border_width*2, w + border_width*2)
    padded[border_width + 1 : h + border_width,
           border_width + 1 : w + border_width] = ref_frame
    
    row, col = block_num
    block_size = size(current_block)[1]
    y, x = block_size*(row), block_size*(col)
    search_area = padded[y + 1 : y+ search_size, x + 1 : x + search_size]
    loss_prev = 1000
    y_moved::Int, x_moved::Int = -1, -1
    for j in 1:(search_size -block_size)
        for i in 1:(search_size - block_size)
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
    h, w = size(ref_frame)
    motion_vectors = zeros(Int, n_rows, n_cols,2)
    for col in 1:n_cols
        for row in 1:n_rows   
            @views motion_vectors[row, col, :] = motion_estimation(ref_frame, current_frame[(row - 1)*n_cols + col , :, :],
            (row - 1,col - 1), n_rows, n_cols, search_size)
        end
    end
    motion_vectors
end