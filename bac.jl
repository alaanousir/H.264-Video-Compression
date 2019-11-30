# The ranges were not hard coded to enable 32-bit systems to work well.
# I let Julia decide what is the default (32 or 64)
# NOTE: top of range is exclusive this is why we use -1 
# NOTE: Vectors are 1-D arrays in Julia
"""
    bac_config{T<:Integer}
    parametric struct holding the Integer ranges
    Those ranges correspong to 0 , 0.25, 0.75, 1.0 
    Integers are used for more vontrol over percision
"""
struct bac_config{T<:Integer}
    top::T # p < 1.0
    first_qtr::T # p = 0.25
    half::T # p = 0.5
    third_qtr::T # p = 0.75
    """
    Constructor to calculate the ranges from the max
    value
    """
    function bac_config{T}(top) where {T<:Integer}
        fq = div(top,4) +1 # first quarter
        # div is used for iteger division
        new(top, fq, 2fq, 3fq)
    end
end

""" Returns the Integer type from config """
bac_type(_::bac_config{T}) where T = T

"""
    bac_encode_state{T<:Integer}
    parametric struct holding the state of the encoder
    holds lower and upper bound currently used 
    And the remembered bits for handling 0.25<= range < 0.75 cases
"""
mutable struct bac_encode_state{T<:Integer}
    low::T
    high::T
    # for outputting multiple bits when in the middle section
    bits_to_follow::T
end


"""Initializes the encoder state with the same type as theconfig"""
init_bac_encode(conf) = bac_encode_state{bac_type(conf)}(0, conf.top, 0)

"""
    Release bit or bits using a later passed output_bit function
    output_bit is not self defined to be modular and enable 
    writing to a buffer or a file by passing a different output_bit 
    function
"""
function bits_plus_follow(bit::Bool, state::bac_encode_state, output_bit)
    output_bit(bit)
    while state.bits_to_follow >0
        output_bit(!bit) 
        state.bits_to_follow -= 1
    end
end

"""
    Perform one BAC_encode encoding step in the given bit 
    T is a parametric type inferred from the type of state
    and conf. Enforces state and config to be the same type
    Args:
        bit::Bool: incoming bit to encode
        state::bac_encode_state{T}: state of the encoder
        conf::bac_config{T}: state of the encoder
        LOP::AbstractFloat: least occuring bit(LOB) probability

"""
function encode_step(bit::Bool, state::bac_encode_state{T},
    conf::bac_config{T}, LOP::AbstractFloat,
    LOB::Bool, output_bit) where T

    range = T(state.high - state.low + 1) # calculate range and ensure type is T
    # Update low and high 
    # if LOB [0, po) * range 
    # if MOB [po, 1) * range 
    state.high = state.low + (if bit != LOB range else round(range*LOP) end) -1
    state.low = state.low + (if bit != LOB round(range*LOP) else 0 end)

    while true
        # Apply Scaling if possible, else return
        if state.high < conf.half
            # if in lower half send LOB
            bits_plus_follow(false, state, output_bit)
        elseif state.low >= conf.half
            # if in lower half send MOB
            bits_plus_follow(true, state, output_bit)
            # Then subtract half
            state.low -= conf.half
            state.high -= conf.half
        elseif state.low >= conf.first_qtr && state.high < conf.third_qtr
            # Middle section
            # Tell state to output an extra Opposite bit
            state.bits_to_follow += 1
            # Then subtract half
            state.low -= conf.first_qtr
            state.high -= conf.first_qtr
        else
            break
        end

        # Scale the range up by 2
        state.low = state.low*2
        state.high = (state.high + 1)*2 - 1
    end
end
"""
    finalize(state::bac_encode_state{T}, conf::bac_config{T}) where T
    sends the final bits
"""
function finalize(state::bac_encode_state{T}, output_bit) where T
    #NOTE: low is guranteed by scaling to be less than half
    # So we need to output two bits either 01 or 10 
    state.bits_to_follow += 1
    if state.low < conf.first_qtr 
        bits_plus_follow(false, state, output_bit)
    else
        bits_plus_follow(true, state, output_bit)
    end
end

# Init the condugration with the system's Unsigned Int
# 2^63 - 1 for 64 bit systems and 2^31 -1 for 32
conf = bac_config{UInt}((typemax(UInt)>>1) - 1) 

"""
    bitstream_bac_encode(data::Union{Vector{Bool}, BitArray{1}})
    encodes a bit stream using Binary Arithmatic Code
    Input:
        data: is either a Vector{Bool} or BitArray{1}
              they are operationally the same in Julia but each element in 
              a bitarray is stored in 1 but while in Vector it is i 8 normally.
              in our case a numpy with dtype Bool will be passed.
"""
function bitstream_bac_encode(data::Union{Vector{Bool}, BitArray{1}})
    # Initialize state using config
    state = init_bac_encode(conf) 

    p1 = sum(data)/length(data) # probability of bit 1 

    LOP = if p1 <= 0.5 p1 else 1.0 - p1 end #Least Occuring Probability

    LOB =  p1 <= 0.5 #Least Occuring Bit

    # Get the Bit Array of the LOP
    binary_LOP = BitArray(digits(UInt(round(LOP*(conf.top+1))), base=2, pad=64))[end:-1:1]

    # Get the Bit Array of the length of data
    bin_len_data = BitArray(digits(UInt(length(data)), base=2, pad=64))[end:-1:1]

    # result BitArray starting with the LOB, LOP and length of data
    res = BitArray([Bool[LOB] ; binary_LOP; bin_len_data])
    
    # define how to output bits; append bits to res
    output_bit(bit::Bool) =  push!(res, bit)
    for bit::Bool ∈ data
        encode_step(bit, state, conf, LOP, LOB, output_bit)
    end
    finalize(state, output_bit)
    res
end



##############################DECODING###################################



"""
    bac_encode_state{T<:Integer}
    parametric struct holding the state of the decoder
    holds lower and upper bound currently used, and the value read from 
    input bits
"""
mutable struct bac_decode_state{T<:Integer}
    low::T
    high::T
    value::T
end

"""
Initializes the encoder state with the same type as the config
"""
init_bac_decode(conf) = bac_decode_state{bac_type(conf)}(0, conf.top, 0)

"""
bitstream_bac_encode(data::Union{Vector{Bool}, BitArray{1}})
decodes a bit stream using Binary Arithmatic Code
Input:
    data: is either a Vector{Bool} or BitArray{1}
          they are operationally the same in Julia but each element in 
          a bitarray is stored in 1 but while in Vector it is i 8 normally.
          in our case a numpy with dtype Bool will be passed.
"""
function bitstream_bac_decode(encoded_data::Union{Vector{Bool}, BitArray{1}})

    LOB = popfirst!(encoded_data) # Get the Least Occuring Bit

    # Calculate the LOP from the binary stream 
    binary_LOP = [popfirst!(encoded_data) for _ ∈ 1:64] # Step 1 Pull the bits
    LOP = sum((UInt64(1) .<< (63:-1:0)).*binary_LOP) / ((typemax(UInt64)>>1))
          # sum the the bits converted to Unsigned Int divided by the maximum of our range

    # Calculate the Length of signal from bitstream
    bin_len_data = [popfirst!(encoded_data) for _ ∈ 1:64] # Step 1 Pull the bits
    bin_len_data = sum((UInt64(1) .<< (63:-1:0)).*bin_len_data) # Step 2 convert to UInt
                    # sum the the bits converted to Unsigned Int
    # Initialize state using config
    state = init_bac_decode(conf)

    # Take in as much bits as our percision can handle
    # The rest of the bits are read bit by bit using scaling
    bits_to_read = if UInt == UInt64 63 else 31 end
    for _ in 1:bits_to_read
        state.value = UInt(2)*state.value + popfirst!(encoded_data)
    end
    # Initialize the result BitArray for
    res = BitArray{1}([])
    while length(res) < bin_len_data
        push!(res, decode_step(encoded_data, state, conf, LOP, LOB))
    end
    res
end


function decode_step(encoded_data::Union{Vector{Bool}, BitArray{1}},
    state::bac_decode_state{T},
    conf::bac_config{T}, LOP::AbstractFloat,
    LOB::Bool) where T
    range = T(state.high - state.low + 1) # calculate range and ensure type is T
    cp = T(state.value - state.low + 1)/range
    bit = cp >= LOP 
    # if LOB [0, po) * range 
    # if MOB [po, 1) * range 
    state.high = state.low + (if bit != LOB range else round(range*LOP) end) -1
    state.low = state.low + (if bit != LOB round(range*LOP) else 0 end)
    while true
        # Apply Scaling if possible, else return
        if state.high < conf.half
            ;
        elseif state.low >= conf.half
            # if in lower half 
            # subtract half
            state.low -= conf.half
            state.high -= conf.half
            state.value -= conf.half
        elseif state.low >= conf.first_qtr && state.high < conf.third_qtr
            # Middle section
            # Then subtract half
            state.low -= conf.first_qtr
            state.high -= conf.first_qtr
            state.value -= conf.first_qtr
        else
            break
        end
        # Scale the range up by 2
        state.low = state.low*2
        state.high = (state.high + 1)*2 - 1
        state.value = 2 * state.value +
         if isempty(encoded_data) 0 else popfirst!(encoded_data) end
    end
    if bit !LOB else LOB end
end
