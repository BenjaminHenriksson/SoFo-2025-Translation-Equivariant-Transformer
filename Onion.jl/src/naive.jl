using Flux
using ConcreteStructs

export NaiveRoPE, NaiveTransformerBlock

struct NaiveRoPE
    theta::Float32
end

Flux.@layer NaiveRoPE trainable=()

function NaiveRoPE(;theta=10000f0)
    return NaiveRoPE(theta)
end

function (rope::NaiveRoPE)(x, positions)
    # x: (head_dim, seq_len, n_heads, batch)
    # positions: (d_coords, seq_len, batch)
    D, S, H, B = size(x)
    d_coords, S_pos, B_pos = size(positions)
    
    @assert D % 6 == 0 "Head dimension must be divisible by 6, but got $D"
    @assert d_coords == 3 "d_coords must be 3 for NaiveRoPE"
    @assert S == S_pos && B == B_pos "Sequence length or batch size mismatch between x and positions"

    num_pairs = D ÷ 2
    T = eltype(x)
    freqs = 1.0f0 ./ (rope.theta .^ (T.(0:2:D-1)[1:num_pairs] ./ D)) # shape (num_pairs,)
    
    pos_indices = Int.((0:num_pairs-1) .% 3 .+ 1)
    selected_pos = positions[pos_indices, :, :]

    angles = reshape(freqs, num_pairs, 1, 1) .* selected_pos
    
    cos_vals = cos.(angles)
    sin_vals = sin.(angles)

    cos_vals = reshape(cos_vals, num_pairs, S, 1, B)
    sin_vals = reshape(sin_vals, num_pairs, S, 1, B)

    x1 = x[1:D÷2, :, :, :]
    x2 = x[D÷2+1:end, :, :, :]
    
    rotated_x = vcat(
        x1 .* cos_vals .- x2 .* sin_vals,
        x2 .* cos_vals .+ x1 .* sin_vals
    )
    return rotated_x
end

@concrete struct NaiveTransformerBlock
    attention
    feed_forward
    attention_norm
    ffn_norm
    rope
end

function NaiveTransformerBlock(
    dim::Int, n_heads::Int, d_coords::Int, n_kv_heads::Int = n_heads, ff_hidden_dim = 4 * dim;
    norm_eps=1f-5, qkv_bias=false, rope_theta=10000f0
)
    @assert d_coords == 3 "d_coords must be 3 for NaiveRoPE"
    head_dim = dim ÷ n_heads
    @assert head_dim % 6 == 0 "Head dimension must be divisible by 6 for NaiveRoPE"

    NaiveTransformerBlock(
        Attention(dim, n_heads, n_kv_heads; qkv_bias=qkv_bias),
        StarGLU(dim, ff_hidden_dim),
        RMSNorm(dim, eps=norm_eps),
        RMSNorm(dim, eps=norm_eps),
        NaiveRoPE(theta=rope_theta)
    )
end

function (block::NaiveTransformerBlock)(x; positions, mask=0)
    h = x + block.attention(block.attention_norm(x); rope=block.rope, mask=mask, positions=positions)
    out = h + block.feed_forward(block.ffn_norm(h))
    return out
end

Flux.@layer NaiveTransformerBlock
