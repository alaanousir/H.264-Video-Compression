get_bits(data, pad) = BitArray(digits(data, base=2, pad=pad))[end:-1:1]

function binarize_mv(vid_mv)
    offset = minimum(vid_mv)
    vid_mv = vid_mv .+ -offset

    mv_size = Int(ceil(log2(maximum(vid_mv))))
    n_frames, n_row, n_col, _ = size(vid_mv)

    # encode the size of each motion vector (in bits)
    # upper limit of 255
    res = get_bits(mv_size, 8)
    offset_bin = get_bits(abs(offset), mv_size)
    n_frames_bin = get_bits(n_frames, 64)
    n_row_bin = get_bits(n_row, 16)
    n_col_bin = get_bits(n_col, 16)
    res = [res ; offset >= 0; offset_bin; n_frames_bin; n_row_bin; n_col_bin]
    println(length(res))
    for f in 1:n_frames for c in 1:n_row for r in 1:n_col
        mv1 = get_bits(vid_mv[f,c,r,1], mv_size)
        mv2 = get_bits(vid_mv[f,c,r,2], mv_size)
        push!(res, mv1..., mv2...)
    end    end    end
    println(length(res))
    res
end
read_bits_to_decimal(bitstream, nbits) = sum((1 .<< ((nbits-1):-1:0)).* 
                                       [popfirst!(bitstream) for _ ∈ 1:nbits])
function debinarize_mv(bitstream)
    mv_size = read_bits_to_decimal(bitstream, 8)
    offset_sign = if popfirst!(bitstream) 1 else -1 end
    offset = read_bits_to_decimal(bitstream, mv_size)
    n_frames = read_bits_to_decimal(bitstream,64)
    n_row = read_bits_to_decimal(bitstream,16)
    n_col = read_bits_to_decimal(bitstream,16)
    println(length(bitstream))
    vid_mv = zeros(Int, n_frames, n_row, n_col, 2) 
    for f in 1:n_frames for c in 1:n_row for r in 1:n_col
        vid_mv[f,c,r,1] = read_bits_to_decimal(bitstream, mv_size)
        vid_mv[f,c,r,2] = read_bits_to_decimal(bitstream, mv_size)
    end    end    end
    vid_mv .+ offset_sign*offset , n_frames, n_row, n_col
end

function binarize_res(quantized_coeff_y, quantized_coeff_cb, quantized_coeff_cr)
    offset = min(quantized_coeff_y..., quantized_coeff_cb... , quantized_coeff_cr...)
    quantized_coeff_y = quantized_coeff_y .+ -offset
    quantized_coeff_cb = quantized_coeff_cb .+ -offset
    quantized_coeff_cr = quantized_coeff_cr .+ -offset

    max_res = max(quantized_coeff_y..., quantized_coeff_cb... , quantized_coeff_cr...)
    res_size = Int(ceil(log2(max_res)))
    @assert res_size==9 "each should occupy 9 bits"

    res = BitArray([offset >= 0; get_bits(abs(offset), res_size)]) # empty BitArray
    for i in 1:size(quantized_coeff_y)[1] for j in 1:size(quantized_coeff_y)[2]
            push!(res, get_bits(quantized_coeff_y[i,j], res_size)...)
    end end
    for i in 1:size(quantized_coeff_cb)[1] for j in 1:size(quantized_coeff_cb)[2]
        push!(res, get_bits(quantized_coeff_cb[i,j], res_size)...)
    end end
    for i in 1:size(quantized_coeff_cr)[1] for j in 1:size(quantized_coeff_cr)[2]
        push!(res, get_bits(quantized_coeff_cr[i,j], res_size)...)
    end end
    res
end

function debinarize_res(bitstream, n_blocks, n_frames)
    res_size = 9
    offset_sign = if popfirst!(bitstream) 1 else -1 end
    offset = read_bits_to_decimal(bitstream, res_size)
    quantized_coeff_y = zeros(Int, n_frames, n_blocks * 16 * 16)
    quantized_coeff_cb = zeros(Int, n_frames, n_blocks * 8 * 8)
    quantized_coeff_cr = zeros(Int, n_frames, n_blocks * 8 * 8)
    for i in 1:size(quantized_coeff_y)[1] for j in 1:size(quantized_coeff_y)[2]
        quantized_coeff_y[i, j] = read_bits_to_decimal(bitstream, res_size)
    end end
    for i in 1:size(quantized_coeff_cb)[1] for j in 1:size(quantized_coeff_cb)[2]
        quantized_coeff_cb[i, j] = read_bits_to_decimal(bitstream, res_size)
    end end
    for i in 1:size(quantized_coeff_cr)[1] for j in 1:size(quantized_coeff_cr)[2]
        quantized_coeff_cr[i, j] = read_bits_to_decimal(bitstream, res_size)
    end end
    quantized_coeff_y .+ offset_sign*offset,
    quantized_coeff_cb .+ offset_sign*offset,
    quantized_coeff_cr .+ offset_sign*offset
end

