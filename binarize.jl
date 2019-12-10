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
    for f in 1:n_frames for c in 1:n_row for r in 1:n_col
        mv1 = get_bits(vid_mv[f,c,r,1], mv_size)
        mv2 = get_bits(vid_mv[f,c,r,2], mv_size)
        push!(res, mv1..., mv2...)
    end    end    end
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
    vid_mv = zeros(Int, n_frames, n_row, n_col, 2) 
    for f in 1:n_frames for c in 1:n_row for r in 1:n_col
        vid_mv[f,c,r,1] = read_bits_to_decimal(bitstream, mv_size)
        vid_mv[f,c,r,2] = read_bits_to_decimal(bitstream, mv_size)
    end    end    end
    vid_mv .+ offset_sign*offset , n_frames, n_row, n_col
end

function binarize_res(quantized_coeff_y, quantized_coeff_cb, quantized_coeff_cr)    
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
    println(offset, '\t', n_frames, '\t', max_res, '\t',res_size)
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

function debinarize_res(bitstream, n_frames)
    res_size = 9
    offset_sign = if popfirst!(bitstream) 1 else -1 end
    offset = read_bits_to_decimal(bitstream, res_size)
    quantized_coeff_y = [Int[] for _ in 1:n_frames]
    quantized_coeff_cb = [Int[] for _ in 1:n_frames]
    quantized_coeff_cr = [Int[] for _ in 1:n_frames]

    function _read_color_helper(arr)
        for i in 1:n_frames
            for j in 1:read_bits_to_decimal(bitstream,64)
                push!(arr[i], read_bits_to_decimal(bitstream, res_size)...)
            end
            arr[i] .+= offset_sign*offset
    end end
    _read_color_helper(quantized_coeff_y) 
    _read_color_helper(quantized_coeff_cb)
    _read_color_helper(quantized_coeff_cr)
    quantized_coeff_y, quantized_coeff_cb, quantized_coeff_cr
end

function binarize_ref_frames(ref_frames::Array{Array{UInt8,2},2})
    ref_frames_y = cat(ref_frames[:,1]...,dims=3)
    ref_frames_cb = cat(ref_frames[:,2]...,dims=3)
    ref_frames_cr = cat(ref_frames[:,3]...,dims=3)
    println(size(ref_frames_y), '\t', size(ref_frames_cb), '\t', size(ref_frames_cr), '\t')
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
function debinarize_ref_frames(bitstream, n_frames, nrow, ncol)
    ref_frames_y = Array{UInt8,3}(undef, n_frames, ncol, nrow)
    ref_frames_cb = Array{UInt8,3}(undef, n_frames, div(ncol,2), div(nrow,2))
    ref_frames_cr = Array{UInt8,3}(undef, n_frames, div(ncol,2), div(nrow,2))
    function _debinarize_ref_helper(arr)
        for i in 1:size(arr)[1] for j in 1:size(arr)[2] for k in 1:size(arr)[3]
            arr[i,j,k] = read_bits_to_decimal(bitstream,8)
    end end end end
    _debinarize_ref_helper(ref_frames_y)
    _debinarize_ref_helper(ref_frames_cb)
    _debinarize_ref_helper(ref_frames_cr)
    (ref_frames_y, ref_frames_cb, ref_frames_cr)
end