# Multidimensional RoPE (STRING) from Schneck et al. 2025:
# "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING".
# Link to paper: https://arxiv.org/abs/2502.02562

@info "Using local STRING.jl (dev env)"

# FOR DEV ONLY
using Random; rng = Xoshiro(0)

@concrete struct STRING
    dim
    d_coords
    n_heads
    thetas
    orthogonal_parameter
end

Flux.@layer STRING

# dim is the dimensionality of the model (head), d_coords is the dimensionality of the position vector (often R^3)
function STRING(dim::Int, d_coords::Int, n_heads::Int)
    @assert iseven(dim) "Dimensionality (dim) must be even for STRING, dim=$dim was given."

    return STRING(
        dim,                                # Dimensionality of head
        d_coords,                           # Dimensionality of token position space
        n_heads,
        randn(rng, Float32, dim ÷ 2, n_heads),
        randn(rng, Float32, dim, dim, n_heads),
    )
end

# Batch‑aware variant – accepts x with shape (seq_len, batch)
# Returns rotation matrices of shape (2, 2, k, heads, seq_len, batch)
function ContinuousRoPE(x::AbstractArray, rope::STRING)
    phase = rope.thetas .* reshape(x, 1, 1, size(x,1), size(x,2))

    c = cos.(phase)
    s = sin.(phase)

    flat = cat(c, -s, s, c; dims=5)                     # (k, heads, S, B, 4)
    rot = reshape(flat, size(c,1), rope.n_heads, size(x,1), size(x,2), 2, 2)
    rot = permutedims(rot, (5,6,1,2,3,4))               # (2,2,k,heads,S,B)
    return rot
end

# Expects `position` with shape (d_coords, seq_len, batch)
# Returns a tensor with shape (dim, dim, seq_len, heads, batch)
function (rope::STRING)(position::AbstractArray)
    @assert ndims(position) == 3 "Position must have shape (d_coords, seq_len, batch)."
    @assert size(position,1) == rope.d_coords "Coordinate dimension mismatch."

    coordinate_sum = dropdims(sum(position; dims=1), dims=1)  # (seq_len, batch)

    # Rotation blocks: (2,2,k,heads,seq_len,batch)
    MultiRope = ContinuousRoPE(coordinate_sum, rope)

    A = rope.orthogonal_parameter                                     # (dim, dim, heads)
    skew = A .- permutedims(A, (2,1,3))
    P = mapslices(exp, skew; dims=(1,2))                               # (dim, dim, heads)
    PT = permutedims(P, (2,1,3))                                       # (dim, dim, heads)

    k = size(MultiRope,3)                                              # number of 2×2 blocks
    S = size(MultiRope,5)
    B = size(MultiRope,6)
    γ = size(PT,2)                                                     # dim

    D = reshape(MultiRope, 2, 2, k, rope.n_heads, S, B, 1)             # α β k h S B 1
    Q = reshape(PT,        1, 2, k, rope.n_heads, 1, 1, γ)             # 1 β k h 1 1 γ
    out = dropdims(sum(D .* Q; dims=2), dims=2)                        # (2,k,h,S,B,γ)

    out = reshape(out, rope.dim, rope.n_heads, S, B, γ)                # (dim,h,S,B,dim)
    out_for_mul = reshape(permutedims(out, (1,3,4,5,2)), rope.dim, S*B*γ, rope.n_heads)
    final = batched_mul(P, out_for_mul)                                # (dim,S*B*γ,h)
    final = reshape(final, rope.dim, S, B, γ, rope.n_heads)            # (dim,S,B,dim,h)
    final = permutedims(final, (1,4,2,5,3))                            # (dim,dim,S,h,B)

    return final
end

@concrete struct STRINGTransformerBlock
    attention
    feed_forward
    attention_norm
    ffn_norm
    stringfield
end

function STRINGTransformerBlock(
    dim::Int, n_heads::Int, d_coords::Int, n_kv_heads::Int = n_heads, ff_hidden_dim = 4 * dim;
    norm_eps=1f-5, qkv_bias=false,
)
    STRINGTransformerBlock(
        Attention(dim, n_heads, n_kv_heads; qkv_bias),
        StarGLU(dim, ff_hidden_dim),
        RMSNorm(dim, eps=norm_eps),
        RMSNorm(dim, eps=norm_eps),
        STRING(Int(dim/n_heads), d_coords, n_heads),
    )
end

function (block::STRINGTransformerBlock)(x; start_pos=1, mask=0, positions=nothing)
    h = x + block.attention(block.attention_norm(x), start_pos, block.stringfield, mask; positions=positions)
    out = h + block.feed_forward(block.ffn_norm(h))
    return out
end

# compat
(block::STRINGTransformerBlock)(x, start_pos, rope=identity, mask=0) =
    block(x; start_pos, rope, mask)

Flux.@layer STRINGTransformerBlock
