module SimpleTransformer

using LinearAlgebra
using Random
using Statistics

# -----------------------------
# Utilities
# -----------------------------

"""Stable soft‑max along a given dimension (no `keepdims` keyword, for broad julia compatibility)."""
function softmax(x::AbstractArray; dims::Integer=1)
    m   = maximum(x; dims=dims)        # Broadcast subtract will stretch `m` automatically.
    ex  = exp.(x .- m)
    ex ./ sum(ex; dims=dims)           # Division broadcasts to match original shape.
end

# -----------------------------
# Rotary Positional Embedding (RoPE)
# -----------------------------

"""Rotary embedding tables holding cos/sin values (Su et al., 2021)."""
struct RotaryEmbedding
    head_dim::Int           # even
    max_len::Int
    cos::Matrix{Float32}    # (max_len, head_dim/2)
    sin::Matrix{Float32}    # (max_len, head_dim/2)
end

function RotaryEmbedding(head_dim::Int; max_len::Int=5000)
    @assert iseven(head_dim) "head_dim must be even for RoPE"
    half      = head_dim ÷ 2
    dim_range = collect(0:half-1)
    inv_freq  = (10000.0f0) .^ (-2 .* Float32.(dim_range) ./ head_dim)  # (half,)
    positions = Float32.(collect(0:max_len-1))                          # (max_len,)
    freqs     = positions * inv_freq'                                   # (max_len, half)
    RotaryEmbedding(head_dim, max_len, cos.(freqs), sin.(freqs))
end

"""Rotate (H,B,S,D) tensor in‑place and return it."""
function apply_rotary(re::RotaryEmbedding, x::Array{Float32,4})
    H,B,S,D = size(x)
    @assert D == re.head_dim
    half = D ÷ 2

    # x → (H,B,S,half,2)  (even,odd)
    x5   = reshape(x, H,B,S,half,2)
    even = @view x5[:,:,:,:,1]
    odd  = @view x5[:,:,:,:,2]

    cos_t = reshape(re.cos[1:S, :], 1,1,S,half)
    sin_t = reshape(re.sin[1:S, :], 1,1,S,half)

    rot_even =  even .* cos_t .- odd .* sin_t
    rot_odd  =  even .* sin_t .+ odd .* cos_t

    x5[:,:,:,:,1] .= rot_even
    x5[:,:,:,:,2] .= rot_odd
    return reshape(x5, H,B,S,D)
end

# -----------------------------
# Layer Normalisation
# -----------------------------

struct LayerNorm
    dim::Int
    γ::Vector{Float32}
    β::Vector{Float32}
    ε::Float32
end

function LayerNorm(dim::Int; ε::Float32=1f-5)
    LayerNorm(dim, ones(Float32,dim), zeros(Float32,dim), ε)
end

function (ln::LayerNorm)(x::Array{Float32,3})
    μ  = mean(x; dims=3)
    σ2 = mean((x .- μ).^2; dims=3)
    normed = (x .- μ) ./ sqrt.(σ2 .+ ln.ε)
    normed .* reshape(ln.γ,1,1,ln.dim) .+ reshape(ln.β,1,1,ln.dim)
end

# -----------------------------
# Multi‑Head Attention (RoPE)
# -----------------------------

struct MultiHeadAttention
    d_model::Int
    num_heads::Int
    head_dim::Int
    rotary::RotaryEmbedding
    W_q::Matrix{Float32}
    W_k::Matrix{Float32}
    W_v::Matrix{Float32}
    W_o::Matrix{Float32}
end

function MultiHeadAttention(d_model::Int, num_heads::Int)
    @assert d_model % num_heads == 0 "d_model must be divisible by num_heads"
    head_dim = d_model ÷ num_heads
    init(m,n) = randn(Float32,m,n) .* sqrt(2/m)
    MultiHeadAttention(
        d_model,
        num_heads,
        head_dim,
        RotaryEmbedding(head_dim),
        init(d_model,d_model),
        init(d_model,d_model),
        init(d_model,d_model),
        init(d_model,d_model)
    )
end

# -----------------------------------------------------------------------------
# Helper reshapes
# -----------------------------------------------------------------------------

split_heads(x, num_heads) = begin
    S,B,D  = size(x)
    head_d = D ÷ num_heads
    x4     = reshape(x, S,B,num_heads,head_d)
    permutedims(x4, (3,2,1,4))  # (H,B,S,Dh)
end

combine_heads(x) = begin
    H,B,S,Dh = size(x)
    x4 = permutedims(reshape(x, H,B,S,Dh), (3,2,1,4))  # (S,B,H,Dh)
    reshape(x4, S,B,H*Dh)                              # (S,B,D)
end

function linear(x::Array{Float32,3}, W::Matrix{Float32})
    S,B,D = size(x)
    x2    = reshape(permutedims(x,(2,1,3)), B*S, D)
    y2    = x2 * W                                   # (B*S, D_out)
    Dout  = size(W,2)
    permutedims(reshape(y2, B,S,Dout), (2,1,3))      # (S,B,D_out)
end

function scaled_dot_product_attention(Q,K,V)
    dk = size(Q,4)
    H,B,S,_ = size(Q)
    out = Array{Float32}(undef,H,B,S,dk)
    inv_sqrt_dk = 1 / sqrt(Float32(dk))
    for h in 1:H, b in 1:B
        scores = (Q[h,b,:,:] * K[h,b,:,:]') * inv_sqrt_dk  # (S,S)
        probs  = softmax(scores; dims=2)
        out[h,b,:,:] = probs * V[h,b,:,:]
    end
    out
end

function (mha::MultiHeadAttention)(q::Array{Float32,3}, k::Array{Float32,3}, v::Array{Float32,3})
    Q = linear(q, mha.W_q)
    K = linear(k, mha.W_k)
    V = linear(v, mha.W_v)

    Qh = split_heads(Q, mha.num_heads)
    Kh = split_heads(K, mha.num_heads)
    Vh = split_heads(V, mha.num_heads)

    Qh = apply_rotary(mha.rotary, Qh)
    Kh = apply_rotary(mha.rotary, Kh)

    attn   = scaled_dot_product_attention(Qh, Kh, Vh)
    concat = combine_heads(attn)
    linear(concat, mha.W_o)
end

# -----------------------------
# Feed‑Forward
# -----------------------------

struct FeedForward
    d_model::Int
    d_ff::Int
    W1::Matrix{Float32}
    b1::Vector{Float32}
    W2::Matrix{Float32}
    b2::Vector{Float32}
end

function FeedForward(d_model::Int, d_ff::Int)
    init(m,n) = randn(Float32,m,n) .* sqrt(2/m)
    FeedForward(
        d_model,
        d_ff,
        init(d_model,d_ff),
        zeros(Float32,d_ff),
        init(d_ff,d_model),
        zeros(Float32,d_model)
    )
end

function (ff::FeedForward)(x::Array{Float32,3})
    S,B,_ = size(x)
    x2 = reshape(permutedims(x,(2,1,3)), B*S, ff.d_model)
    h  = max.(0, x2*ff.W1 .+ ff.b1')
    y2 = h*ff.W2 .+ ff.b2'
    permutedims(reshape(y2, B,S,ff.d_model), (2,1,3))
end

# -----------------------------
# Encoder Block
# -----------------------------

struct EncoderBlock
    mha::MultiHeadAttention
    ff::FeedForward
    norm1::LayerNorm
    norm2::LayerNorm
end

function EncoderBlock(d_model::Int, num_heads::Int, d_ff::Int)
    EncoderBlock(
        MultiHeadAttention(d_model,num_heads),
        FeedForward(d_model,d_ff),
        LayerNorm(d_model),
        LayerNorm(d_model)
    )
end

function (eb::EncoderBlock)(x::Array{Float32,3})
    attn_out = eb.mha(x,x,x)
    x        = eb.norm1(x .+ attn_out)
    ff_out   = eb.ff(x)
    eb.norm2(x .+ ff_out)
end

# -----------------------------
# Transformer Encoder
# -----------------------------

struct TransformerEncoder
    layers::Vector{EncoderBlock}
end

function TransformerEncoder(n_layers::Int, d_model::Int, num_heads::Int, d_ff::Int)
    TransformerEncoder([EncoderBlock(d_model,num_heads,d_ff) for _ in 1:n_layers])
end

function (enc::TransformerEncoder)(src::Array{Float32,3})
    x = src
    for blk in enc.layers
        x = blk(x)
    end
    x
end

# -----------------------------
# Demo
# -----------------------------

"""Minimal forward‑pass sanity check (RoPE)."""
function demo()
    Random.seed!(42)
    S,B,D   = 12, 4, 64
    H,DFF,L = 8, 256, 2
    enc     = TransformerEncoder(L, D, H, DFF)
    dummy   = randn(Float32, S,B,D)
    out     = enc(dummy)
    @assert size(out) == (S,B,D)
    println("Output shape: ", size(out))
end

demo()

end # module
