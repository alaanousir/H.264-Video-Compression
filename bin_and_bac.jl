include("binarize.jl")
include("bac.jl")
function encode_bin_bac(vid_mv,predictedPerRef, no_frames,
                        quantized_coeffs, ref_frames)
    res = [get_bits(no_frames, 64); get_bits(predictedPerRef, 16)]
    push!(res, binarize_mv(vid_mv)...)
    push!(res, binarize_res(quantized_coeffs)...)
    push!(res, binarize_ref_frames(ref_frames)...)
    bitstream_bac_encode(res)
end

function decode_bin_bac(bitstream)
    binarized = bitstream_bac_decode(bitstream)
    no_frames = read_bits_to_decimal(binarized, 64)
    predictedPerRef = read_bits_to_decimal(binarized, 16)
    n_predicted_frames = no_frames - Int(ceil(no_frames/predictedPerRef))
    vid_mv, n_row, n_col = debinarize_mv(binarized, n_predicted_frames)
    residuals = debinarize_res(binarized, n_predicted_frames)
    ref_frames = debinarize_ref_frames(binarized, no_frames - n_predicted_frames,
                                       n_row*16, n_col*16)
    vid_mv, residuals, ref_frames
end