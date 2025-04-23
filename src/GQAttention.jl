#Scaled dot product attention
function sdpa(xq::AbstractArray{T}, xk::AbstractArray{T}, xv::AbstractArray{T}, head_dim::Int, mask = 0) where T
    A = softmax(batched_mul(batched_transpose(xk), xq) / sqrt(T(head_dim)) .+ mask; dims=1)
    return batched_mul(xv, A)
end

"""
    Attention(dim::Int, n_heads::Int, n_kv_heads=n_heads; qkv_bias=false)

Attention layer that supports both self-attention and cross-attention (as in Llama3).

# Self-attention example
```julia
dim = 64
n_heads = 8
n_kv_heads = 4

attn = Attention(dim, n_heads, n_kv_heads)
output = attn(x)  # Self-attention
```

# Cross-attention example
```julia
output = attn(query, key, value)  # Cross-attention
```
"""
mutable struct Attention{DA, DB, DC, DD}
    wq::DA
    wk::DB
    wv::DC
    wo::DD
    dim::Int
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
end

Flux.@layer Attention

function Attention(dim::Int, n_heads::Int, n_kv_heads=n_heads; qkv_bias=false)
    head_dim = dim ÷ n_heads
    Attention(
        Dense(dim => n_heads * head_dim, bias=qkv_bias),
        Dense(dim => n_kv_heads * head_dim, bias=qkv_bias),
        Dense(dim => n_kv_heads * head_dim, bias=qkv_bias),
        Dense(n_heads * head_dim => dim, bias=false),
        dim,
        n_heads,
        n_kv_heads,
        head_dim
    )
end

repeat_kv(x::AbstractArray, n_rep::Int) = isone(n_rep) ? x : repeat(x, 1, n_rep, 1, 1)

# Backward compatibility method for self-attention with existing interface
function (attn::Attention)(x::AbstractArray{T}, start_pos::Integer=1, rope=nothing, mask=0) where T
    return attn(x, nothing, nothing, start_pos, rope, mask)
end

function (attn::Attention)(query::AbstractArray{T}, key::Union{Nothing, AbstractArray{T}}=nothing, value::Union{Nothing, AbstractArray{T}}=nothing, start_pos::Integer=1, rope=nothing, mask=0) where T
    # If key and value are not provided, use query (self-attention)
    key = isnothing(key) ? query : key
    value = isnothing(value) ? key : value
    
    _, q_seqlen, q_batch = size(query)
    _, k_seqlen, k_batch = size(key)
    
    xq = attn.wq(query)
    xk = attn.wk(key)
    xv = attn.wv(value)
    
    xq = reshape(xq, (attn.head_dim, attn.n_heads, q_seqlen, q_batch))
    xk = reshape(xk, (attn.head_dim, attn.n_kv_heads, k_seqlen, k_batch))
    xv = reshape(xv, (attn.head_dim, attn.n_kv_heads, k_seqlen, k_batch))
    
    xq = permutedims(xq, (1,3,2,4))
    xk = permutedims(xk, (1,3,2,4))
    xv = permutedims(xv, (1,3,2,4))
    
    if rope isa RoPE
        xq, xk = rope(xq), rope(xk)
    end
    
    # Update if cache is configured with seq_length > 0
    #xk, xv = update!(attn.cache, start_pos, xk, xv)
    
    # Repeat keys and values for multi-query attention if needed
    xk = repeat_kv(xk, attn.n_heads ÷ attn.n_kv_heads)
    xv = repeat_kv(xv, attn.n_heads ÷ attn.n_kv_heads)
    
    xq_for_attn = reshape(xq, attn.head_dim, :, attn.n_heads * q_batch)
    xk_for_attn = reshape(xk, attn.head_dim, :, attn.n_heads * k_batch)
    xv_for_attn = reshape(xv, attn.head_dim, :, attn.n_heads * k_batch)
    
    output = sdpa(xq_for_attn, xk_for_attn, xv_for_attn, attn.head_dim, mask)
    
    e_output = reshape(output, (attn.head_dim, q_seqlen, attn.n_heads, q_batch))
    p_output = permutedims(e_output, (1,3,2,4)) 
    r_output = reshape(p_output, (attn.n_heads * attn.head_dim, q_seqlen, q_batch))
    
    proj = attn.wo(r_output)
    return proj
end

struct TransformerBlock{A,F,AN,FN}
    attention::A
    feed_forward::F
    attention_norm::AN
    ffn_norm::FN
end

"""
    TransformerBlock(dim::Int, n_heads::Int, n_kv_heads::Int = n_heads, ff_hidden_dim = 4 * dim; norm_eps=1f-5, qkv_bias=false)
    TransformerBlock{Attention,FeedForward,AttentionNorm,FeedForwardNorm}

Transformer block for GQAttention (as in Llama3). No KV caching (see Jjama3.jl for KV caching).
    
```julia
dim = 64
n_heads = 8
n_kv_heads = 4
seqlen = 10

rope = RoPE(dim ÷ n_heads, 1000)
t = TransformerBlock(dim, n_heads, n_kv_heads)

h = randn(Float32, dim, seqlen, 1)

#Use without a mask:
h = t(h, 1, rope[1:seqlen])

#Use with a causal mask:
mask = Onion.causal_mask(h)
h = t(h, 1, rope[1:seqlen], mask)
```
"""
function TransformerBlock(
    dim::Int, n_heads::Int, n_kv_heads::Int = n_heads, ff_hidden_dim = 4 * dim;
    norm_eps=1f-5, qkv_bias=false
)
    TransformerBlock(
        Attention(dim, n_heads, n_kv_heads; qkv_bias),
        StarGLU(dim, ff_hidden_dim),
        RMSNorm(dim, eps=norm_eps),
        RMSNorm(dim, eps=norm_eps)
    )
end

function (block::TransformerBlock)(x, start_pos, rope, mask = 0)
    h = x + block.attention(block.attention_norm(x), start_pos, rope, mask)
    out = h + block.feed_forward(block.ffn_norm(h))
    return out
end

Flux.@layer TransformerBlock



struct AdaTransformerBlock{A,F,AN,FN}
    attention::A
    feed_forward::F
    attention_norm::AN
    ffn_norm::FN
end

function AdaTransformerBlock(
    dim::Int, cond_dim::Int, n_heads::Int, n_kv_heads::Int = n_heads, ff_hidden_dim = 4 * dim;
    norm_eps=1f-5, qkv_bias=false
)
    AdaTransformerBlock(
        Attention(dim, n_heads, n_kv_heads; qkv_bias),
        StarGLU(dim, ff_hidden_dim),
        AdaLN(dim, cond_dim),
        AdaLN(dim, cond_dim)
    )
end

function (block::AdaTransformerBlock)(x, cond, rope, mask)
    h = x + block.attention(block.attention_norm(x, cond), 0, rope, mask)
    out = h + block.feed_forward(block.ffn_norm(h, cond))
    return out
end

Flux.@layer AdaTransformerBlock




function causal_mask(h::AbstractArray{T}) where T<:AbstractFloat
    Flux.ChainRulesCore.ignore_derivatives() do
        dim, seqlen, batch = size(h)
        mask = similar(h, seqlen, seqlen)
        mask .= T(-Inf)
        mask = tril(mask, -1) #This is swapped because we're using the slightly more efficient dim setup
        return mask
    end
end
