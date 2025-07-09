"""
    StarGLU(dim::Int, ff_hidden_dim::Int; act=Flux.swish)

Gated Linear Unit with flexible activation function (default: `swish`, making it a SwiGLU layer as used in Llama3).

```julia
l = StarGLU(6, 8)
h = randn(Float32, 6, 10, 1)
h = l(h)
```
"""
struct StarGLU{W, F}
    w1::W
    w2::W
    w3::W
    act::F
end

Flux.@layer StarGLU

function StarGLU(dim::Int, ff_hidden_dim::Int; act=Flux.swish)
    StarGLU(
        Dense(dim => ff_hidden_dim, bias=false),
        Dense(ff_hidden_dim => dim, bias=false),
        Dense(dim => ff_hidden_dim, bias=false),
        act
    )
end

(ff::StarGLU)(x) = ff.w2(ff.act(ff.w1(x)) .* ff.w3(x))