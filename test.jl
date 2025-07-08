using Pkg; Pkg.activate("."); Pkg.instantiate()

using Test, Onion

module STRINGBackup
    include(joinpath(@__DIR__, "Onion.jl", "src", "STRING_backup.jl"))
end

@testset "STRING Batched vs Non-Batched" begin
    dim = 8
    d_coords = 3
    seq_len = 4
    batch = 2

    rope_batched = Onion.STRING(dim, d_coords)
    rope_nonbatched = STRINGBackup.STRING(dim, d_coords)

    positions = rand(Float32, d_coords, seq_len, batch)
    batched_out = rope_batched(positions)

    expected = Array{Float32}(undef, dim, seq_len, batch, dim)
    for b in 1:batch
        for s in 1:seq_len
            expected[:, s, b, :] = rope_nonbatched(positions[:, s, b])
        end
    end

    @test isapprox(batched_out, expected; rtol=1e-5, atol=1e-5)
end