function binarize_mv(vid_mv)
    offset = minimum(vid_mv)
    vid_mv = vid_mv .+ -offset

    mv_size = Int(ceil(log2(maximum(vid_mv))))
    n_frames, n_cols, n_rows, _ = size(vid_mv)

    # encode the size of each motion vector (in bits)
    # upper limit of 255
    res = BitArray(digits(mv_size, base=2, pad=8))[end:-1:1]
    offset_bin = BitArray(digits(abs(offset), base=2, pad=mv_size))[end:-1:1]
    n_frames_bin = BitArray(digits(n_frames, base=2, pad=64))[end:-1:1]
    n_cols_bin = BitArray(digits(n_cols, base=2, pad=16))[end:-1:1]
    n_rows_bin = BitArray(digits(n_rows, base=2, pad=16))[end:-1:1]
    res = [res ; offset >= 0; offset_bin; n_frames_bin; n_cols_bin; n_rows_bin]
    println(length(res))
    for f in 1:n_frames for c in 1:n_cols for r in 1:n_rows
        mv1 = BitArray(digits(vid_mv[f,c,r,1], base=2, pad=mv_size))[end:-1:1]
        mv2 = BitArray(digits(vid_mv[f,c,r,2], base=2, pad=mv_size))[end:-1:1]
        push!(res, mv1..., mv2...)
    end    end    end
    println(length(res))
    res
end

function debinarize_mv(bitstream)
    println(length(bitstream))
    mv_size = sum((1 .<< (7:-1:0)).*[popfirst!(bitstream) for _ ∈ 1:8])
    offset_sign = if popfirst!(bitstream) 1 else -1 end
    offset = sum((1 .<< (mv_size-1:-1:0)).*[popfirst!(bitstream) for _ ∈ 1:mv_size])
    n_frames = sum((1 .<< (63:-1:0)).*[popfirst!(bitstream) for _ ∈ 1:64])
    n_cols = sum((1 .<< (15:-1:0)).*[popfirst!(bitstream) for _ ∈ 1:16])
    n_rows = sum((1 .<< (15:-1:0)).*[popfirst!(bitstream) for _ ∈ 1:16])
    println(length(bitstream))
    vid_mv = zeros(Int, n_frames, n_cols, n_rows, 2) 
    for f in 1:n_frames for c in 1:n_cols for r in 1:n_rows
        vid_mv[f,c,r,1] = sum((1 .<< (mv_size-1:-1:0)).*[popfirst!(bitstream) for _ ∈ 1:mv_size])
        vid_mv[f,c,r,2] = sum((1 .<< (mv_size-1:-1:0)).*[popfirst!(bitstream) for _ ∈ 1:mv_size])
    end    end    end
    vid_mv .+ offset_sign*offset
end

