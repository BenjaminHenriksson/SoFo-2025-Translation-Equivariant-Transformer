module Onion

using Flux, LinearAlgebra

include("shared.jl")
include("AdaLN.jl")
include("RMSNorm.jl")
include("DyT.jl")
include("StarGLU.jl")
include("GQAttention.jl")
include("RoPE.jl")
include("UNet.jl")
include("FlexibleUNet.jl")

export
    #shared:
    glut,
    #layers:
    AdaLN,
    RMSNorm,
    DyT,
    StarGLU,
    GQAttention,
    RoPE,
    TransformerBlock,
    Attention
    # UNet components:
    GaussianFourierProjection,
    TimeEmbedding,
    ResidualBlock,
    EncoderBlock,
    DecoderBlock,
    Bottleneck,
    FlexibleUNet,
    # UNet helper functions:
    reverse_tuple,
    process_encoders,
    process_decoders
end
