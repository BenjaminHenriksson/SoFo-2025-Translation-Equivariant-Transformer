# Multidimensional RoPE (STRING) from Schneck et al. 2025:
# "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING".
# Link to paper: https://arxiv.org/abs/2502.02562

@info "Using local STRING.jl (dev env)"

# FOR DEV ONLY
using Random;rng = Xoshiro(0)

struct STRING
    dim::Int
    d_coords::Int
    thetas
    orthogonal_parameter
end

#Flux.@layer STRING

# dim is the dimensionality of the model (head), d_coords is the dimensionality of the position vector (often R^3)
function STRING(dim::Int, d_coords::Int)
    @assert iseven(dim) "Dimensionality (dim) must be even for STRING, dim=$dim was given."

    return STRING(
        dim,                                # Dimensionality of head
        d_coords,                           # Dimensionality of token position space
        rand(rng, Float32, dim ÷ 2),        # Thetas (FOR DEV ONLY)
        rand(rng, Float32, dim, dim),       # Orthogonal parameter
    )
end

# Batch‑aware variant  –  accepts x with shape (seq_len, batch)
# Returns rotation matrices of shape (2, 2, k, seq_len, batch)
function ContinuousRoPE(x::AbstractArray, rope::STRING)
    # Phase: (k, S, B)  ←  broadcast multiply
    phase = reshape(rope.thetas, :, 1, 1) .* reshape(x, 1, size(x,1), size(x,2))

    c = cos.(phase)                                 # (k,S,B)
    s = sin.(phase)

    # Assemble rotation blocks (2,2,k,S,B) without loops
    rot = similar(c, Float32, 2, 2, size(c,1), size(c,2), size(c,3))
    rot[1,1,:,:,:] .= c
    rot[1,2,:,:,:] .= -s
    rot[2,1,:,:,:] .= s
    rot[2,2,:,:,:] .=  c
    return rot
end

# -------------------------------------------------------------------
# STRING forward pass – now batch aware
# -------------------------------------------------------------------
# Expects `position` with shape (d_coords, seq_len, batch)
# Returns a tensor with shape (dim, seq_len, batch, dim)
function (rope::STRING)(position::AbstractArray)
    @assert ndims(position) == 3 "Position must have shape (d_coords, seq_len, batch)."
    @assert size(position,1) == rope.d_coords "Coordinate dimension mismatch."

    # Sum over spatial coordinates → (seq_len, batch)
    coordinate_sum = dropdims(sum(position; dims=1), dims=1)

    # Rotation blocks: (2, 2, k, seq_len, batch)
    MultiRope = ContinuousRoPE(coordinate_sum, rope)

    # Orthogonal matrix from eq. 9
    A  = rope.orthogonal_parameter
    P  = exp(A - A')
    PT = P'                                          # (dim, dim)

    k   = size(MultiRope,3)                          # number of 2×2 blocks
    S   = size(MultiRope,4)                          # seq_len
    B   = size(MultiRope,5)                          # batch
    γ   = size(PT,2)                                 # dim

    # Align β‑index (second dim) for contraction
    D = reshape(MultiRope, 2, 2, k, S, B, 1)         # α β k S B 1
    Q = reshape(PT,        1, 2, k, 1, 1, γ)         # 1 β k 1 1 γ

    # Contract over β without explicit loops
    out = dropdims(sum(D .* Q; dims=2), dims=2)      # (2, k, S, B, γ)

    # Merge α & k → dim
    out = reshape(out, rope.dim, S, B, γ)            # (dim, S, B, γ)

    # Left‑multiply by P for each (S,B) slice
    out2d = reshape(out, rope.dim, :)                # flatten trailing dims
    final = P * out2d                                # (dim, S*B*γ)
    final = reshape(final, rope.dim, S, B, γ)        # (dim, S, B, dim)

    return final
end
