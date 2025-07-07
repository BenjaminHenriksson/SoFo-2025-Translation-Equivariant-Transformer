using SparseArrays  # built‐in stdlib
using LinearAlgebra # for cos / sin

struct STRING
    dim
    d_coords
    thetas
    orthogonal_parameter
end

Flux.@layer STRING

# dim is the dimensionality of the model (head), d_coords is the dimensionality of the position vector (often R^3) 
function STRING(dim::Int, d_coords::Int)
    @assert iseven(dim) "Dimensionality (dim) must be even for RoPE, dim=$dim was given."
    # d_coords must divide dim?

    return STRING( 
        dim, # Dimensionality of head
        d_coords, # Dimensionality of token position space
        rand(Float32, dim ÷ 2), # Thetas
        rand(Float32, dim, dim), # Orthogonal paramter
    )
end

# eq. 2/3 STRING paper
function OLD_ContinuousRoPE(x::Number, params::STRING)
    rot(x, θ) = reshape([cos(x*θ), -sin(x*θ), sin(x*θ),  cos(x*θ)], 2, 2)

    mats = [rot(x, params.thetas[i]) for i in 1:params.dim ÷ 2]

    BD_sparse = blockdiag(sparse.(mats)...) # block-diagonal sparse matrix
    BD_dense  = Matrix(BD_sparse) # densify matrix again

    return BD_dense
end

"""
    qdq_block(Q, blocks) -> S

Compute the congruence transform

    S = Q * diagm(blocks...) * Q'

in **O(½ n³) flops** where `blocks` is a `Vector` of small square
matrices (normally all `2×2`, with an optional trailing `1×1` if `n` is odd).

The implementation:

* touches each 2-column block of `Q` exactly once;
* allocates only an `n×n` output plus a single `n×2` scratch per iteration;
* is written entirely with differentiable tensor operations, so Zygote can
  take gradients through it without custom rules.
"""
function qdq_block(Q::AbstractMatrix, blocks::AbstractVector)
    n = size(Q, 1)
    @assert size(Q, 2) == n           "Q must be square"
    @assert sum(size.(blocks, 1)) == n "Block sizes must add up to n"

    S   = zeros(eltype(Q), n, n)      # result
    col = 1                           # current block offset

    @views for B in blocks            # loop over 2×2 (or 1×1) blocks
        k  = size(B, 1)               # block size (2 or 1)
        Qi = Q[:, col:col+k-1]        # (n × k)  — ith block of Q
        Wi = Qi * B                   # (n × k)  — 4 n k flops
        S  += Wi * Qi'                # rank-k update, 2 n² flops
        col += k
    end
    return S
end

# eq. 2/3 STRING paper
function ContinuousRoPE(x::Number, params::STRING)
    rot(x, θ) = reshape([cos(x*θ), -sin(x*θ), sin(x*θ),  cos(x*θ)], 2, 2)

    # construct rotation matrices
    mats = [rot(x, params.thetas[i]) for i in 1:params.dim ÷ 2]

    return mats
end


# pure unoptimized implementation of STRING 
function (rope::STRING)(position::AbstractArray)
    @assert length(position) == rope.d_coords "Coordinate vector dimensionality must match STRING."
    
    OLD_MultiRoPE = Matrix{Float32}(I, rope.dim, rope.dim)
    
    coordinate_sum = sum(position)

    # eq. 5 STRING paper
    MultiRope = ContinuousRoPE(coordinate_sum, rope) 
    
    for i in 1:rope.d_coords    
        OLD_MultiRoPE *= OLD_ContinuousRoPE(position[i], rope) 
    end

    println(MultiRope)
    println(OLD_MultiRoPE)

    # eq. 9 STRING paper
    A = rope.orthogonal_parameter
    P = exp(A - A')
    output = qdq_block(P, MultiRope)

    print(isapprox(output, P * OLD_MultiRoPE * P'))

    return output
end
