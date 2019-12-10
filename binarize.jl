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
    n_row_bin = get_bits(n_row, 16)
    n_col_bin = get_bits(n_col, 16)
    res = [res ; offset >= 0; offset_bin; n_row_bin; n_col_bin]
    for f in 1:n_frames for c in 1:n_row for r in 1:n_col
        mv1 = get_bits(vid_mv[f,c,r,1], mv_size)
        mv2 = get_bits(vid_mv[f,c,r,2], mv_size)
        push!(res, mv1..., mv2...)
    end    end    end
    res
end
function read_bits_to_decimal(bitstream, nbits, pos)
    sum((1 .<< ((nbits-1):-1:0)).* [bitstream[i] for i ∈ pos:(pos+nbits-1)])
end

function debinarize_mv(bitstream, n_frames, pos)
    mv_size = read_bits_to_decimal(bitstream, 8, pos)
    pos += 8
    offset_sign = if bitstream[pos] 1 else -1 end
    pos +=1
    offset = read_bits_to_decimal(bitstream, mv_size, pos)
    pos += mv_size
    n_row = read_bits_to_decimal(bitstream,16, pos)
    pos += 16
    n_col = read_bits_to_decimal(bitstream,16, pos)
    pos += 16
    vid_mv = zeros(Int, n_frames, n_row, n_col, 2) 
    for f in 1:n_frames for c in 1:n_row for r in 1:n_col
        vid_mv[f,c,r,1] = read_bits_to_decimal(bitstream, mv_size, pos)
        pos += mv_size
        vid_mv[f,c,r,2] = read_bits_to_decimal(bitstream, mv_size, pos)
        pos += mv_size
    end    end    end
    vid_mv .+ offset_sign*offset , n_row, n_col, pos
end

function binarize_res(quantized_coeffs)
    quantized_coeff_y, quantized_coeff_cb, quantized_coeff_cr = quantized_coeffs
    offset = min((quantized_coeff_y...)...
                ,(quantized_coeff_cb...)... 
                ,(quantized_coeff_cr...)...)
    n_frames = length(quantized_coeff_y)
    for i in 1:n_frames
        quantized_coeff_y[i] .+= -offset
        quantized_coeff_cb[i] .+= -offset
        quantized_coeff_cr[i] .+= -offset
    end

    max_res = max((quantized_coeff_y...)...
                ,(quantized_coeff_cb...)... 
                ,(quantized_coeff_cr...)...)
    res_size = Int(ceil(log2(max_res)))
    @assert res_size==9 "each should occupy 9 bits"
    res = BitArray([offset >= 0; get_bits(abs(offset), res_size)])

    function _write_color_helper(arr)
        for i in 1:n_frames
            frame_len = length(arr[i])
            push!(res, get_bits(frame_len, 64)...)
             for j in 1:frame_len
                push!(res, get_bits(arr[i][j], res_size)...)
    end end end
    _write_color_helper(quantized_coeff_y)
    _write_color_helper(quantized_coeff_cb)
    _write_color_helper(quantized_coeff_cr)
    res
end

function debinarize_res(bitstream, n_frames, pos)
    res_size = 9
    offset_sign = if bitstream[pos] 1 else -1 end
    pos+=1
    offset = read_bits_to_decimal(bitstream, res_size,pos)
    pos += res_size
    quantized_coeff_y = [Int[] for _ in 1:n_frames]
    quantized_coeff_cb = [Int[] for _ in 1:n_frames]
    quantized_coeff_cr = [Int[] for _ in 1:n_frames]

    function _read_color_helper(arr)
        for i in 1:n_frames
            s = read_bits_to_decimal(bitstream,64, pos)
            pos += 64
            for j in 1:s
                push!(arr[i], read_bits_to_decimal(bitstream, res_size,pos)...)
                pos += res_size
            end
            arr[i] .+= offset_sign*offset
    end end
    _read_color_helper(quantized_coeff_y) 
    _read_color_helper(quantized_coeff_cb)
    _read_color_helper(quantized_coeff_cr)
    (quantized_coeff_y, quantized_coeff_cb, quantized_coeff_cr), pos
end

function binarize_ref_frames(ref_frames::Array{Array{UInt8,2},2})
    ref_frames_y = cat(ref_frames[:,1]...,dims=3)
    ref_frames_cb = cat(ref_frames[:,2]...,dims=3)
    ref_frames_cr = cat(ref_frames[:,3]...,dims=3)
    res = BitArray(Bool[])
    function _binarize_ref_helper(arr)
        for i in 1:size(arr)[3] for j in 1:size(arr)[1] for k in 1:size(arr)[2]
                push!(res, get_bits(arr[j,k,i], 8)...)
        end end end
    end
    _binarize_ref_helper(ref_frames_y)
    _binarize_ref_helper(ref_frames_cb)
    _binarize_ref_helper(ref_frames_cr)
    res
end
function debinarize_ref_frames(bitstream, n_frames, nrow, ncol, pos)
    ref_frames_y = Array{UInt8,3}(undef, n_frames, nrow, ncol)
    ref_frames_cb = Array{UInt8,3}(undef, n_frames, div(nrow,2), div(ncol,2))
    ref_frames_cr = Array{UInt8,3}(undef, n_frames, div(nrow,2), div(ncol,2))
    function _debinarize_ref_helper(arr)
        for i in 1:size(arr)[1] for j in 1:size(arr)[2] for k in 1:size(arr)[3]
            arr[i,j,k] = read_bits_to_decimal(bitstream,8, pos)
            pos += 8
    end end end end
    _debinarize_ref_helper(ref_frames_y)
    _debinarize_ref_helper(ref_frames_cb)
    _debinarize_ref_helper(ref_frames_cr)
    (ref_frames_y, ref_frames_cb, ref_frames_cr), pos
end