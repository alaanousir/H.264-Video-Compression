include("binarize.jl")
include("bac.jl")
function encode_bin_bac(vid_mv,predictedPerRef, no_frames,
                        quantized_coeffs, ref_frames)
    res = [get_bits(no_frames, 64); get_bits(predictedPerRef, 16)]
    bin_mv = binarize_mv(vid_mv)
    bin_res = binarize_res(quantized_coeffs)
    bin_ref = binarize_ref_frames(ref_frames)
    bitstream_bac_encode(vcat(res,bin_mv,bin_res,bin_ref))
end

function decode_bin_bac(bitstream)
    binarized = bitstream_bac_decode(bitstream)
    pos = 1
    println("debaced")
    no_frames = read_bits_to_decimal(binarized, 64, pos)
    pos +=64
    predictedPerRef = read_bits_to_decimal(binarized, 16, pos)
    pos +=16
    n_predicted_frames = no_frames - Int(ceil(no_frames/predictedPerRef))
    println("starting mvs")
    vid_mv, n_row, n_col, pos = debinarize_mv(binarized, n_predicted_frames, pos)
    println("starting mvs")
    residuals, pos = debinarize_res(binarized, n_predicted_frames, pos)
    println("starting mvs")
    ref_frames, pos = debinarize_ref_frames(binarized, no_frames - n_predicted_frames,
                                       n_row*16, n_col*16, pos)
    println(pos)
    println(binarized)
    @assert pos == length(binarized)                                       
    vid_mv, residuals, ref_frames
end