struct RoPE{A<:AbstractArray}
    cos::A
    sin::A
end

Flux.@functor RoPE
Flux.trainable(::RoPE) = ()

Base.getindex(rope::RoPE, i) = RoPE(rope.cos[:, i, :, :], rope.sin[:, i, :, :])

@views function (rope::RoPE)(x)
    half = size(x, 1) ÷ 2
    x1 = view(x, 1:half, :, :, :)
    x2 = view(x, half + 1:2 * half, :, :, :)
    return vcat(
        x1 .* rope.cos .- x2 .* rope.sin,
        x2 .* rope.cos .+ x1 .* rope.sin,
    )
end

struct LearnableRoPE{A<:AbstractVector}
    theta::A
end

Flux.@functor LearnableRoPE

function LearnableRoPE(dim::Int; theta=10000f0)
    LearnableRoPE(param(fill(theta, dim ÷ 2)))
end

function Base.getindex(lr::LearnableRoPE, r::UnitRange{<:Integer})
    half = length(lr.theta)
    dim = 2 * half
    exponents = (0:2:dim - 1) ./ dim
    freqs = 1f0 ./ (lr.theta .^ exponents)
    T = eltype(lr.theta)
    pos = T.(r .- 1)
    freqs_complex = cis.(pos .* transpose(freqs))
    cos = reshape(permutedims(real(freqs_complex), (2, 1)), (half, length(r), 1, 1))
    sin = reshape(permutedims(imag(freqs_complex), (2, 1)), (half, length(r), 1, 1))
    return RoPE(cos, sin)
end
