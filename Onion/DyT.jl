struct DyT{T <: AbstractFloat, V <: AbstractVector{T}}
    alpha::V   # α
    weight::V  # γ
    bias::V    # β
end

Flux.@layer DyT

"""
    DyT(dim::Integer; init_alpha::T = 0.5f0)

Make a Dynamic Tanh (DyT) layer for normalizing the input tensor.

See [Transformers without Normalization](https://arxiv.org/abs/2503.10622) for more details.
"""
DyT(dim::Integer; init_alpha::T = 0.5f0) where T = DyT(ones(T, 1) .* init_alpha, ones(T, dim), zeros(T, dim))

(dyt::DyT)(x) = @. dyt.weight * tanh(dyt.alpha * x) + dyt.bias
