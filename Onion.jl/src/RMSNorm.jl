"""
    RMSNorm(dim::Int; eps::T=1f-5)

Root Mean Square Layer Normalization. As used in Llama3.
"""
struct RMSNorm{T<:AbstractFloat,W<:AbstractVector{T}}
    weight::W
    eps::T
end

Flux.@layer RMSNorm

RMSNorm(dim::Int; eps::T=1f-5) where T = RMSNorm(ones(T, dim), eps)

function (norm::RMSNorm)(x)
    rms = sqrt.(sum(abs2, x, dims=1) / size(x, 1) .+ norm.eps)
    return x .* (norm.weight ./ rms)
end
