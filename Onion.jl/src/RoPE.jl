"""
    RoPE(dim::Int, max_length; theta::T=10000f0)
    

Rotary Position Embeddings (as in Llama3).
    
```julia
dim = 64
n_heads = 8
n_kv_heads = 4
seqlen = 10

t = TransformerBlock(dim, n_heads, n_kv_heads)
h = randn(Float32, dim, seqlen, 1)

rope = RoPE(dim ÷ n_heads, 1000)
h = t(h, 1, rope[1:seqlen]) #Note the subsetting to match seqlen
```
"""
struct RoPE{A<:AbstractArray}
    cos::A
    sin::A
end

Flux.@layer RoPE trainable=()

Base.getindex(rope::RoPE, i) = RoPE(rope.cos[:,i,:,:], rope.sin[:,i,:,:])

function RoPE(dim::Int, end_pos::Int; theta::T=10000f0, start_pos=0) where T
    freqs = 1f0 ./ (theta .^ (T.(0:2:dim-1)[1:dim÷2] ./ dim))
    freqs_complex = cis.(T.(start_pos:end_pos-1) * freqs')
    cos = permutedims(real(freqs_complex), (2, 1))  # (head_dim/2, seq_len)
    sin = permutedims(imag(freqs_complex), (2, 1))
    cos = reshape(cos, (dim÷2, end_pos - start_pos, 1, 1))
    sin = reshape(sin, (dim÷2, end_pos - start_pos, 1, 1))
    return RoPE(cos, sin)
end

# Note about Huggingface weights and rotary embeddings:
# https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509
# Use this one if you're using the Hugging Face weights.
function (rope::RoPE)(x)
    head_dim = size(x, 1)
    x1 = x[1:head_dim÷2, :, :, :]
    x2 = x[head_dim÷2+1:end, :, :, :]
    return vcat(  
        x1 .* rope.cos .- x2 .* rope.sin,
        x2 .* rope.cos .+ x1 .* rope.sin
    )
end

# * Multidimensional RoPE from Schneck et al. 2025:
#   "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING".
#   Link to paper: https://arxiv.org/abs/2502.02562
#
# * This is equivalent to the original RoPE (Su et al. 2023) in eq. 34, but
#   with each token's position vector m ∈ ℜ^n and learnable θ ∈ ℜ^(n × d).

# Permutation from Su et al. 2023 (RoFormer), eq. 34 sine vector
# Currently causes errors:
# BoundsError: attempt to access 8×30×8×10 Array{Float32, 4} at index [1:2:7, 1:30] (when feeding batched position vectors)
# DimensionMismatch: arrays could not be broadcast to a common size: a has axes Base.OneTo(8) and b has axes Base.OneTo(10) (when naïvely broadcasting)
function pairflip(X::AbstractArray)
    @assert iseven(size(X, 1)) # dimensionality of embedding vector must be even
    
    # Magic code to compute (x1, x2, x3, x4, ...) -> (-x2, x1, -x4, x3, ...)
    org_dims = size(X)
    X = reshape(X, size(X, 1), :)
    X_odd = X[1:2:end, :]
    X_even = X[2:2:end, :]
    Y_even = reshape(X_odd, 1, size(X_odd)...)
    Y_odd = reshape(-X_even, 1, size(X_even)...)
    Y = reshape(cat(Y_odd, Y_even, dims=1), org_dims...)
    
    return Y
end

# Ensure current file is loaded when using / developing Onion.jl
println("Using local RoPE.jl (dev env)")

# Struct with learnable matrices for multidimensional RoPE
struct MultiDimRoPE{A}
    Thetas::A
    FreeMatrix::A
end

Flux.@layer MultiDimRoPE

# dim is the dimensionality of the model (head), d_coords is the dimensionality of the position vector (often R^3) 
function MultiDimRoPE(dim::Int, d_coords::Int)
    @assert iseven(dim) "Dimensionality (dim) must be even for RoPE, dim=$dim was given."
    
    # Thetas is learnt in transposed form
    
    # BAD HARD CODED SOLUTION FOR TESTING:
    return MultiDimRoPE( rand(Float32, dim, d_coords), rand(Float32, dim, dim) )
    # return MultiDimRoPE( rand(Float32, 8, d_coords), rand(Float32, 8, 8) )
end

# embedding tensor shapes: (head_dim, seqlen, n_heads, batch)
function (rope::MultiDimRoPE)(x::AbstractArray, positions::AbstractArray)
    batchdims = size(x)[2:end] # (seqlen, n_heads, batch)
    
    # size(x) = (8, 2400)
    x = reshape(x, :, prod(batchdims))
    
    # Broadcast positions across the head dimension if needed
    pos_dims = size(positions)
    if ndims(positions) == 3
        # (d_coords, seqlen, batch) -> (d_coords, seqlen, 1, batch)
        positions = reshape(positions, pos_dims[1], pos_dims[2], 1, pos_dims[3])
    elseif ndims(positions) == 4
        # keep as is
    else
        error("positions must have 3 or 4 dimensions")
    end

    @assert pos_dims[2] == batchdims[1] "position sequence length must match x"
    @assert pos_dims[end] == batchdims[end] "position batch size must match x"

    if size(positions,3) == 1 && batchdims[2] > 1
        positions = repeat(positions, 1, 1, batchdims[2], 1)
    else
        @assert size(positions,3) == batchdims[2] "positions head dimension must match x or be 1"
    end

    positions = reshape(positions, size(positions,1), prod(batchdims))
 
    R = batched_mul(rope.Thetas, positions)

    # size(R_cis) = (8, 2400)
    R_cis = cis.(R)
 
    # size(x_perm) = (8, 2400)
    x_perm = pairflip(x)
 
    # Other options for generating P: Cayley STRING, Circulant STRING
    P = exp((rope.FreeMatrix - rope.FreeMatrix') .* 0.5)
 
    P_adjoint = reshape(P', size(P)..., 1)
    P_adjoint = repeat(P_adjoint, 1, 1, prod(batchdims)) 
    # size(P_adjoint) = (32, 32, 2400)
    
    # size(((x .* real.(R_cis)) .+ (x_perm .* imag.(R_cis)))) = (8, 2400)
    x = batched_vec(P_adjoint, ((x .* real.(R_cis)) .+ (x_perm .* imag.(R_cis))))
    return reshape(x, size(x, 1), batchdims...)
end