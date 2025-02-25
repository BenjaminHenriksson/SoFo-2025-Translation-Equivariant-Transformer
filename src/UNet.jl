"""
    GaussianFourierProjection(embed_dim::Int, scale::T=32.0f0)

Creates a Gaussian Fourier feature projection for time embeddings. Used in diffusion models.

# Arguments
- `embed_dim::Int`: Embedding dimension. Should be even.
- `scale::T=32.0f0`: Scaling factor for the random weights.
"""
struct GaussianFourierProjection{T,S<:AbstractArray{T}}
    W::S
    scale::T
end

Flux.@layer GaussianFourierProjection
Flux.trainable(::GaussianFourierProjection) = (;)

function GaussianFourierProjection(embed_dim::Int, scale::T=32.0f0) where T<:AbstractFloat
    W = randn(T, embed_dim ÷ 2) .* scale
    GaussianFourierProjection(W, scale)
end

function (layer::GaussianFourierProjection{T})(t) where T
    t_proj = t' .* layer.W * T(2π)
    vcat(sin.(t_proj), cos.(t_proj))
end

"""
    TimeEmbedding(embed_dim::Int, num_classes::Int, embedding_dim::Int)

Creates time and optional class embeddings for diffusion models.

# Arguments
- `embed_dim::Int`: Output dimension for time embeddings
- `num_classes::Int`: Number of classes for conditional generation
- `embedding_dim::Int`: Dimension for class embeddings

# Examples
```julia
time_emb = TimeEmbedding(256, 10, 128)
t = randn(Float32, 16)
labels = rand(1:10, 16)
h = time_emb(t, labels)
```
"""
struct TimeEmbedding{P,D,E,L}
    proj::P
    dense::D
    embedding::E
    label_dense::L
end

function TimeEmbedding(embed_dim::Int, num_classes::Int, embedding_dim::Int)
    TimeEmbedding(
        GaussianFourierProjection(embed_dim),
        Chain(Dense(embed_dim, embed_dim, swish)),
        Flux.Embedding(num_classes, embedding_dim),
        Chain(Dense(embedding_dim, embed_dim, swish))
    )
end

Flux.@layer TimeEmbedding

function (te::TimeEmbedding)(t)
    h = te.proj(t)
    te.dense(h)
end

function (te::TimeEmbedding)(t, labels)
    h_time = te.proj(t)
    h_time = te.dense(h_time)

    h_label = te.embedding(labels)
    h_label = te.label_dense(h_label)

    h_time .+ h_label
end

"""
    ResidualBlock(channels::Int; kernel_size=3, time_emb=false, emb_dim=256)

A ResNet-style residual block with optional time embeddings.

# Arguments
- `channels::Int`: Number of input and output channels
- `kernel_size=3`: Size of convolutional kernel
- `time_emb=false`: Whether to use time embeddings
- `emb_dim=256`: Dimension of time embeddings

# Examples
```julia
rb = ResidualBlock(64, time_emb=true, emb_dim=256)
h = randn(Float32, 32, 32, 64, 1)
t = randn(Float32, 256, 1)
h = rb(h, t)
```
"""
struct ResidualBlock{C1,C2,N1,N2,D}
    conv1::C1
    conv2::C2
    norm1::N1
    norm2::N2
    time_mlp::D
end

Flux.@layer ResidualBlock

function ResidualBlock(channels::Int; kernel_size=3, time_emb=false, emb_dim=256)
    return ResidualBlock(
        Conv((kernel_size, kernel_size), channels=>channels, pad=1),
        Conv((kernel_size, kernel_size), channels=>channels, pad=1),
        BatchNorm(channels),
        BatchNorm(channels),
        time_emb ? Dense(emb_dim, channels, swish) : identity
    )
end

function (rb::ResidualBlock)(x)
    identity = x
    out = rb.conv1(x)
    out = rb.norm1(out)
    out = relu.(out)
    out = rb.conv2(out)
    out = rb.norm2(out)
    relu.(out + identity)
end

function (rb::ResidualBlock)(x, t)
    identity = x
    out = rb.conv1(x)
    out = rb.norm1(out)
    out = relu.(out)
    t_proj = rb.time_mlp(t)
    t_emb = reshape(t_proj, (1, 1, size(t_proj)...))
    out = out .+ t_emb
    out = rb.conv2(out)
    out = rb.norm2(out)
    relu.(out + identity)
end

"""
    EncoderBlock(in_channels::Int, out_channels::Int; time_emb=false, emb_dim=256)

An encoder block for UNet architecture with optional time embeddings.

# Arguments
- `in_channels::Int`: Number of input channels
- `out_channels::Int`: Number of output channels
- `time_emb=false`: Whether to use time embeddings
- `emb_dim=256`: Dimension of time embeddings

# Examples
```julia
enc = EncoderBlock(3, 64, time_emb=true, emb_dim=256)
h = randn(Float32, 32, 32, 3, 1)
t = randn(Float32, 256, 1)
skip, h = enc(h, t)
```
"""
struct EncoderBlock{C,N,R,P,D}
    conv::C
    norm::N
    residual::R
    pool::P
    time_mlp::D
end

Flux.@layer EncoderBlock

function EncoderBlock(in_channels::Int, out_channels::Int; time_emb=false, emb_dim=256)
    return EncoderBlock(
        Conv((3, 3), in_channels=>out_channels, pad=1),
        BatchNorm(out_channels),
        ResidualBlock(out_channels, time_emb=time_emb, emb_dim=emb_dim),
        MaxPool((2,2)),
        time_emb ? Dense(emb_dim, out_channels, swish) : identity
    )
end

function (eb::EncoderBlock)(x)
    x = eb.conv(x)
    x = eb.norm(x)
    x = relu.(x)
    x = eb.residual(x)
    return x, eb.pool(x)
end

function (eb::EncoderBlock)(x, t)
    x = eb.conv(x)
    x = eb.norm(x)
    x = relu.(x)
    t_proj = eb.time_mlp(t)
    t_emb = reshape(t_proj, (1, 1, size(t_proj)...))
    x = x .+ t_emb
    x = eb.residual(x, t)
    return x, eb.pool(x)
end

"""
    Bottleneck(channels::Int; time_emb=false, emb_dim=256)

A bottleneck block for UNet architecture with optional time embeddings.

# Arguments
- `channels::Int`: Number of input and output channels
- `time_emb=false`: Whether to use time embeddings
- `emb_dim=256`: Dimension of time embeddings

# Examples
```julia
bn = Bottleneck(256, time_emb=true, emb_dim=256)
h = randn(Float32, 8, 8, 256, 1)
t = randn(Float32, 256, 1)
h = bn(h, t)
```
"""
struct Bottleneck{C1,C2,N1,N2,R,D}
    conv1::C1
    conv2::C2
    norm1::N1
    norm2::N2
    residual::R
    time_mlp::D
end

Flux.@layer Bottleneck

function Bottleneck(channels::Int; time_emb=false, emb_dim=256)
    return Bottleneck(
        Conv((3, 3), channels=>channels, pad=1),
        Conv((3, 3), channels=>channels, pad=1),
        BatchNorm(channels),
        BatchNorm(channels),
        ResidualBlock(channels, time_emb=time_emb, emb_dim=emb_dim),
        time_emb ? Dense(emb_dim, channels, swish) : identity
    )
end

function (b::Bottleneck)(x)
    x = b.conv1(x)
    x = b.norm1(x)
    x = relu.(x)
    x = b.conv2(x)
    x = b.norm2(x)
    b.residual(x)
end

function (b::Bottleneck)(x, t)
    x = b.conv1(x)
    x = b.norm1(x)
    x = relu.(x)
    t_proj = b.time_mlp(t)
    t_emb = reshape(t_proj, (1, 1, size(t_proj)...))
    x = x .+ t_emb
    x = b.conv2(x)
    x = b.norm2(x)
    b.residual(x, t)
end

"""
    DecoderBlock(in_channels::Int, out_channels::Int; time_emb=false, emb_dim=256)

A decoder block for UNet architecture with optional time embeddings.

# Arguments
- `in_channels::Int`: Number of input channels
- `out_channels::Int`: Number of output channels
- `time_emb=false`: Whether to use time embeddings
- `emb_dim=256`: Dimension of time embeddings

# Examples
```julia
dec = DecoderBlock(256, 128, time_emb=true, emb_dim=256)
h = randn(Float32, 8, 8, 256, 1)
skip = randn(Float32, 16, 16, 128, 1)
t = randn(Float32, 256, 1)
h = dec(h, skip, t)
```
"""
struct DecoderBlock{U,C,N,R,D}
    upsample::U
    conv::C
    norm::N
    residual::R
    time_mlp::D
end

Flux.@layer DecoderBlock

function DecoderBlock(in_channels::Int, out_channels::Int; time_emb=false, emb_dim=256)
    return DecoderBlock(
        Upsample(:bilinear, scale=2),
        Conv((3, 3), in_channels * 2=>out_channels, pad=1),
        BatchNorm(out_channels),
        ResidualBlock(out_channels, time_emb=time_emb, emb_dim=emb_dim),
        time_emb ? Dense(emb_dim, out_channels, swish) : identity
    )
end

function (db::DecoderBlock)(x, skip_connection)
    x = db.upsample(x)
    x = cat(x, skip_connection, dims=3)
    x = db.conv(x)
    x = db.norm(x)
    x = relu.(x)
    db.residual(x)
end

function (db::DecoderBlock)(x, skip_connection, t)
    x = db.upsample(x)
    x = cat(x, skip_connection, dims=3)
    x = db.conv(x)
    x = db.norm(x)
    x = relu.(x)
    t_proj = db.time_mlp(t)
    t_emb = reshape(t_proj, (1, 1, size(t_proj)...))
    x = x .+ t_emb
    db.residual(x, t)
end

"""
    ResUNet(;
        in_channels=3,
        out_channels=3,
        channels=[64, 128, 256],
        time_embedding=false,
        num_classes=0,
        embedding_dim=128,
        time_emb_dim=256
    )

A residual UNet architecture with optional time and class embeddings, useful for diffusion models and image-to-image tasks.

# Arguments
- `in_channels=3`: Number of input channels
- `out_channels=3`: Number of output channels
- `channels=[64, 128, 256]`: Channel dimensions at each level
- `time_embedding=false`: Whether to use time embeddings
- `num_classes=0`: Number of class labels for conditional generation
- `embedding_dim=128`: Dimension for class embeddings
- `time_emb_dim=256`: Dimension for time embeddings

# Examples
```julia
model = ResUNet(
    in_channels=3,
    out_channels=3,
    channels=[64, 128, 256],
    time_embedding=true,
    num_classes=10,
    embedding_dim=128,
    time_emb_dim=256
)
x = randn(Float32, 32, 32, 3, 1)
t = randn(Float32, 1)
labels = [5]
y = model(x, t, labels)
```
"""
struct ResUNet{E1,E2,E3,B,D3,D2,D1,FC,T}
    enc1::E1
    enc2::E2
    enc3::E3
    bottleneck::B
    dec3::D3
    dec2::D2
    dec1::D1
    final_conv::FC
    time_embed::T
end

Flux.@layer ResUNet

function ResUNet(;
    in_channels=3,
    out_channels=3,
    channels=[64, 128, 256],
    time_embedding=false,
    num_classes=0,
    embedding_dim=128,
    time_emb_dim=256
)
    return ResUNet(
        EncoderBlock(in_channels, channels[1], time_emb=time_embedding, emb_dim=time_emb_dim),
        EncoderBlock(channels[1], channels[2], time_emb=time_embedding, emb_dim=time_emb_dim),
        EncoderBlock(channels[2], channels[3], time_emb=time_embedding, emb_dim=time_emb_dim),
        Bottleneck(channels[3], time_emb=time_embedding, emb_dim=time_emb_dim),
        DecoderBlock(channels[3], channels[2], time_emb=time_embedding, emb_dim=time_emb_dim),
        DecoderBlock(channels[2], channels[1], time_emb=time_embedding, emb_dim=time_emb_dim),
        DecoderBlock(channels[1], channels[1], time_emb=time_embedding, emb_dim=time_emb_dim),
        Conv((1, 1), channels[1]=>out_channels),
        time_embedding ? TimeEmbedding(time_emb_dim, num_classes, embedding_dim) : identity
    )
end

function (model::ResUNet)(x)
    skip1, x = model.enc1(x)
    skip2, x = model.enc2(x)
    skip3, x = model.enc3(x)

    x = model.bottleneck(x)

    x = model.dec3(x, skip3)
    x = model.dec2(x, skip2)
    x = model.dec1(x, skip1)

    model.final_conv(x)
end

function (model::ResUNet)(x, t::T) where T <: AbstractArray
    t = model.time_embed(t)

    skip1, x = model.enc1(x, t)
    skip2, x = model.enc2(x, t)
    skip3, x = model.enc3(x, t)

    x = model.bottleneck(x, t)

    x = model.dec3(x, skip3, t)
    x = model.dec2(x, skip2, t)
    x = model.dec1(x, skip1, t)

    model.final_conv(x)
end

function (model::ResUNet)(x, t::T, labels::L) where {T <: AbstractArray, L <: AbstractArray}
    t = model.time_embed(t, labels)

    skip1, x = model.enc1(x, t)
    skip2, x = model.enc2(x, t)
    skip3, x = model.enc3(x, t)

    x = model.bottleneck(x, t)

    x = model.dec3(x, skip3, t)
    x = model.dec2(x, skip2, t)
    x = model.dec1(x, skip1, t)

    model.final_conv(x)
end
