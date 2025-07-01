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
#   For now without the orthogonal matrix P in eq. 9.
#
# * This is equivalent to the original RoPE (Su et al. 2023) in eq. 34, but
#   with each token's position vector m ∈ ℜ^n and learnable θ ∈ ℜ^(n × d).


# Argument 'i' is the index of the token *as fed into the MultiDimRoPE* 
# e.g.  tokens postioned at (1,2,3), (4,5,6), (7,8,9) come it to the model,
#       i = 1 ⇒ token positioned at (1,2,3)
#       i = 2 ⇒ token positioned at (4,5,6)
#       etc.
# That means the order in which tokens are fed into the model is important! 
# (If new points are generated: append, don't push, new points)

# x = token embedding
# Can be optimized for static positions using matmuls
#function MultidimensionalRoPE(dim::Int, x::AbstractArray, pos::AbstractArray, positions::AbstractMatrix, thetas::ThetaMatrix)

# Permutation from Su et al. 2023 (RoFormer), eq. 34 sine vector

function pairflip(X::AbstractArray)
    @assert iseven(size(X, 1)) # d_embedding is even
    #println("X: ", X, ", size(X): ", size(X))
    org_dims = size(X)
    X = reshape(X, size(X, 1), :)
    X_odd = X[1:2:end, :]
    X_even = X[2:2:end, :]
    Y_even = reshape(X_odd, 1, size(X_odd)...)
    Y_odd = reshape(-X_even, 1, size(X_even)...)
    Y = reshape(cat(Y_odd, Y_even, dims=1), org_dims...)
    return Y
end
# BoundsError: attempt to access 8×30×8×10 Array{Float32, 4} at index [1:2:7, 1:30]

# DimensionMismatch: arrays could not be broadcast to a common size: a has axes Base.OneTo(8) and b has axes Base.OneTo(10)

println("dev env")

# Initialize ThetaMatrix struct
struct MultiDimRoPE{A}
    Thetas::A
    FreeMatrix::A
end

Flux.@layer MultiDimRoPE

function MultiDimRoPE(dim::Int, d_coords::Int)
    @assert iseven(dim) "Dimensionality (dim) must be even for RoPE"
    
    # Thetas is learnt in transposed form
    return MultiDimRoPE( rand(Float32, dim, d_coords), rand(Float32, dim ÷ 2, dim ÷ 2) )
end

function (rope::MultiDimRoPE)(x::AbstractArray, positions::AbstractArray)
    R = batched_mul(rope.Thetas, positions)
    R_cis = cis.(R)

    # Becomes different
    x_perm = pairflip(x)

    P = exp((rope.FreeMatrix - rope.FreeMatrix') .* 0.5)

    return batched_mul(((x .* real.(R_cis)) .+ (x_perm .* imag.(R_cis))), P)
end

