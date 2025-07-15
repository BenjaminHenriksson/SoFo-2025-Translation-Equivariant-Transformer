using Pkg; Pkg.activate(".")
using Test, Onion, Flux

@testset "MultiHeadSTRING Gradients" begin
    dim = 8
    n_heads = 2
    d_coords = 3
    seq_len = 4
    batch = 1

    mhstring = Onion.MultiHeadSTRING(dim, n_heads, d_coords)
    positions = rand(Float32, d_coords, seq_len, batch)

    # simple loss: sum of outputs
    loss() = sum(mhstring(positions))

    gs = gradient(Flux.params(mhstring)) do
        loss()
    end

    # ensure gradients exist for each STRING head
    for h in mhstring.string_heads
        @test haskey(gs, h)
    end
end
