# Multidimensional RoPE (STRING) from Schneck et al. 2025:
# "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING".
# Link to paper: https://arxiv.org/abs/2502.02562

@info "Using local STRING_backup.jl (dev env)"

# FOR DEV ONLY
using Flux, Random;rng = Xoshiro(0)

struct STRING
    dim::Int
    d_coords::Int
    thetas
    orthogonal_parameter
end

Flux.@layer STRING

# dim is the dimensionality of the model (head), d_coords is the dimensionality of the position vector (often R^3) 
function STRING(dim::Int, d_coords::Int)
    @assert iseven(dim) "Dimensionality (dim) must be even for STRING, dim=$dim was given."
    
    return STRING( 
        dim, # Dimensionality of head
        d_coords, # Dimensionality of token position space
        rand(rng, Float32, dim ÷ 2), # Thetas, FOR DEV ONLY: rand(rng, Float32, dim ÷ 2)
        rand(rng, Float32, dim, dim), # Orthogonal paramter
    )
end

# eq. 2/3 STRING paper
function ContinuousRoPE(x::Real, rope::STRING)
    # Everything here stays Float32

    phase = x .* rope.thetas # Vectorised multiply
    c, s = cos.(phase), sin.(phase) # two N-length vectors
    
    # Initialize block diagonal matrix with shape (2, 2, number of blocks)
    out = Array{eltype(c),3}(undef, 2, 2, rope.dim ÷ 2)

    # Construct rotation matrices
    out[1, 1, :] = c
    out[1, 2, :] = -s
    out[2, 1, :] = s
    out[2, 2, :] = c

    return out
end

# Implementation of STRING 
function (rope::STRING)(position::AbstractArray)
    @assert length(position) == rope.d_coords "Coordinate vector dimensionality must match STRING instance."
    
    # Optimization consequence of eq. 5, STRING paper
    coordinate_sum = sum(position)

    # eq. 5 STRING paper, diagonal matrix of 2×2 blocks stored in third dimension: (2, 2, dim÷2)
    MultiRope = ContinuousRoPE(coordinate_sum, rope) 
    
    # Orthogonal matrix in eq. 9 STRING paper
    A = rope.orthogonal_parameter
    P = exp(A - A')
    P_transpose = P'

    # The following is *a way* of multiplying a (2, 2, nblocks) block diagonal matrix, might be optimizable
    nblocks = size(MultiRope,3) # number of blocks RoPE = dim÷2
    ncols = size(P,2) # number of columns in P = dim
    
    # 4-D views that align the contraction index (β) as dimension 2
    D = reshape(MultiRope, 2, 2, nblocks, 1)        # α  β  k  1
    Q = reshape(P_transpose, 1, 2, nblocks, ncols)  # 1  β  k  γ

    # Element-wise multiply, then sum over β to contract it:
    # out[α,k,γ] = Σ_β  D[α,β,k,1] * Q[1,β,k,γ]
    output = dropdims(sum(D .* Q; dims=2), dims=2) # (2,k,ncols)
    output = reshape(output, rope.dim, rope.dim) # back to (n, ncols)
    
    # This should be equivalent to solely "return output" in the attention layer, but 
    # "P * output" is required here for translational invariance test cases to hold
    return P * output
end
