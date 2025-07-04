#Scaled dot product attention
function sdpa(xq::AbstractArray{T}, xk::AbstractArray{T}, xv::AbstractArray{T}, head_dim::Int, mask = 0) where T
    A = softmax(batched_mul(batched_transpose(xk), xq) / sqrt(T(head_dim)) .+ mask; dims=1)
    return batched_mul(xv, A)
end

#For the case where the mask differs for each element in a batch
function sdpa(xq::AbstractArray{T}, xk::AbstractArray{T}, xv::AbstractArray{T}, head_dim::Int, mask::AbstractArray{T, 3}) where T
    d1,d2 = size(xk, 2), size(xq, 2)
    A = softmax(reshape(reshape(batched_mul(batched_transpose(xk), xq) / sqrt(T(head_dim)), d1, d2, :, size(mask, 3)) .+ reshape(mask, d1, d2, 1, :), d1, d2, :), dims=1)
    return batched_mul(xv, A)
end

"""
    self_att_padding_mask(padmask; T = Float32)

Takes a sequence-level `padmask` (ie. length-by-batch, where 0 indicates a padded position) and returns a (non-causal) self-attention mask
that is length-by-length-by-batch and which prevents information flow from padded positions to unpadded positions.
"""
function self_att_padding_mask(padmask; T = Float32)
    pm = T.(padmask)
    mask = log.(clamp.(reshape(pm, 1, size(pm)...) .* reshape(pm, size(pm,1), 1, size(pm,2)) .+ Diagonal(similar(padmask, size(padmask,1)) .= 1), 0, 1))
    return mask
end

"""
    cross_att_padding_mask(padmask, other_dim; T = Float32)

Takes a sequence-level `padmask` and a dimension `other_dim` and returns a cross-attention mask that is length-by-other_dim-by-batch.
This prevents information flow from padded `key` positions to any `query` positions (but ignores padding in the `query` positions, because nothing should flow out of those).
"""
function cross_att_padding_mask(padmask, other_dim; T = Float32)
    pm = T.(padmask)
    return log.(reshape(pm, size(pm,1), 1, size(pm,2)) .* (similar(pm, 1, other_dim, size(pm,2)) .= 1))
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
function (attn::Attention)(x::AbstractArray{T}, start_pos::Integer=1, mask=0; rope=nothing, x_pos::AbstractArray{T}=nothing) where T
    return attn(x, x, start_pos, mask, rope=rope, x_pos=x_pos)
end

function (attn::Attention)(x_query::AbstractArray{T}, x_key::AbstractArray{T}, start_pos::Integer=1, mask=0; rope=nothing, x_pos::AbstractArray{T}=nothing) where T
    _, q_seqlen, q_batch = size(x_query)
    _, k_seqlen, k_batch = size(x_key)
    
    xq = attn.wq(x_query)
    xk = attn.wk(x_key)
    xv = attn.wv(x_key)
    
    xq = reshape(xq, (attn.head_dim, attn.n_heads, q_seqlen, q_batch))
    xk = reshape(xk, (attn.head_dim, attn.n_kv_heads, k_seqlen, k_batch))
    xv = reshape(xv, (attn.head_dim, attn.n_kv_heads, k_seqlen, k_batch))
    
    xq = permutedims(xq, (1,3,2,4))
    xk = permutedims(xk, (1,3,2,4)) # (head_dim, seqlen, n_heads, batch)
    xv = permutedims(xv, (1,3,2,4))
    
    if rope isa RoPE
        xq, xk = rope(xq), rope(xk) 
    elseif rope isa MultiDimRoPE
        @assert x_pos isa AbstractArray "Positions vectors for each embedding must be loaded to use MultiDimRoPE."
        
        shift = randn(Float32, 3)
        xq_s, xk_s = rope(xq, x_pos .+ shift), rope(xk, x_pos .+ shift) 

        xq, xk = rope(xq, x_pos), rope(xk, x_pos) 
        
        @assert !isapprox(xq, xq_s) "MultiDimRoPE didn't detect initial position shifted"
    end  

    # Update if cache is configured with seq_length > 0
    #xk, xv = update!(attn.cache, start_pos, xk, xv)
    
    # Repeat keys and values for multi-query attention if needed
    xk = repeat_kv(xk, attn.n_heads ÷ attn.n_kv_heads)
    xv = repeat_kv(xv, attn.n_heads ÷ attn.n_kv_heads)
    
    xk_s = repeat_kv(xk_s, attn.n_heads ÷ attn.n_kv_heads)
    
    xq_for_attn = reshape(xq, attn.head_dim, :, attn.n_heads * q_batch)
    xk_for_attn = reshape(xk, attn.head_dim, :, attn.n_heads * k_batch)
    xv_for_attn = reshape(xv, attn.head_dim, :, attn.n_heads * k_batch)
    
    xqs_for_attn = reshape(xq_s, attn.head_dim, :, attn.n_heads * q_batch)
    xks_for_attn = reshape(xk_s, attn.head_dim, :, attn.n_heads * q_batch)
    
    # Type issue for unknown reasons, converts from float64 to float32 
    # probably from MultiDimRoPE
    xq_for_attn = convert(Array{Float32, 3}, xq_for_attn)
    xk_for_attn = convert(Array{Float32, 3}, xk_for_attn)
    
    xqs_for_attn = convert(Array{Float32, 3}, xqs_for_attn)
    xks_for_attn = convert(Array{Float32, 3}, xks_for_attn)
    
    output = sdpa(xq_for_attn, xk_for_attn, xv_for_attn, attn.head_dim, mask)
    output_shifted = sdpa(xqs_for_attn, xks_for_attn, xv_for_attn, attn.head_dim, mask)
    
    @assert isapprox(output, output_shifted; atol=1e-5, rtol=1e-5) "Output and shifted output not equal, mean differnce = ", println(sum(abs, output-output_shifted) / length(output))
    println("Translationally invariant (good)")
    
    e_output = reshape(output, (attn.head_dim, q_seqlen, attn.n_heads, q_batch))
    p_output = permutedims(e_output, (1,3,2,4)) 
    r_output = reshape(p_output, (attn.n_heads * attn.head_dim, q_seqlen, q_batch))
    
    proj = attn.wo(r_output)
    return proj
end

struct TransformerBlock{A,F,AN,FN,R}
    attention::A
    feed_forward::F
    attention_norm::AN
    ffn_norm::FN
    rope::R
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
    norm_eps=1f-5, qkv_bias=false, rope=nothing# Rope function argument
)
    #if !(rope isa nothing)
    @assert rope isa MultiDimRoPE "MultiDimRoPE not loaded." 
    #end
    TransformerBlock(
        Attention(dim, n_heads, n_kv_heads; qkv_bias),
        StarGLU(dim, ff_hidden_dim),
        RMSNorm(dim, eps=norm_eps),
        RMSNorm(dim, eps=norm_eps),
        rope, # Must be initialized in function call
    ) 
end

# rope argument made redundant due to block.rope
# Maybe set new version to work with revised TransformerBlocks?
function (block::TransformerBlock)(x, start_pos, rope; x_pos = nothing, mask = 0)
    h = x + block.attention(block.attention_norm(x), start_pos, mask, rope=block.rope, x_pos=x_pos)
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
