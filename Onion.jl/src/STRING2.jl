using SparseArrays  # built‐in stdlib
using LinearAlgebra # for cos / sin

struct STRING
    dim
    d_coords
    thetas
    orthogonal_parameter
end

using Random;rng = Xoshiro(0)

#Flux.@layer STRING

# dim is the dimensionality of the model (head), d_coords is the dimensionality of the position vector (often R^3) 
function STRING(dim::Int, d_coords::Int)
    @assert iseven(dim) "Dimensionality (dim) must be even for RoPE, dim=$dim was given."
    # d_coords must divide dim?

    return STRING( 
        dim, # Dimensionality of head
        d_coords, # Dimensionality of token position space
        rand(rng, Float32,dim ÷ 2), # Thetas, RNG PARAMS FOR DEV ONLY
        rand(rng, Float32, dim, dim), # Orthogonal paramter
    )
end

# eq. 2/3 STRING paper
function ContinuousRoPE(x::Number, params::STRING)
    rot(x, θ) = reshape([cos(x*θ), -sin(x*θ), sin(x*θ),  cos(x*θ)], 2, 2)

    mats = [rot(x, params.thetas[i]) for i in 1:params.dim ÷ 2]

    BD_sparse = blockdiag(sparse.(mats)...) # block-diagonal sparse matrix
    BD_dense  = Matrix(BD_sparse) # densify matrix again

    return BD_dense
end

# pure unoptimized implementation of STRING 
function (rope::STRING)(position::AbstractArray)
    @assert length(position) == rope.d_coords "Coordinate vector dimensionality must match STRING."
    
    MultiRoPE = Matrix{Float32}(I, rope.dim, rope.dim)
    
    for i in 1:rope.d_coords    
        MultiRoPE *= ContinuousRoPE(position[i], rope) 
    end
    
    # eq. 9 STRING paper
    A = rope.orthogonal_parameter
    P = exp(A - A')
    output = P * MultiRoPE * P'

    return output
end
