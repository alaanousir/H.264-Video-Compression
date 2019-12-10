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
    println(length(bitstream))
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
    vid_mv .+ offset_sign*offset
end

