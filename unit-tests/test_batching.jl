# REQUIRES SEEDED RNG GENERATORS FOR BOTH INSTANCES
using Pkg; Pkg.activate("."); Pkg.instantiate()

using Test, Onion, Random

module STRINGBackup
    include(joinpath(@__DIR__, "Onion.jl", "src", "STRING_backup.jl"))
end

@testset "STRING Batched vs Non-Batched" begin
    dim = 8
    d_coords = 3
    seq_len = 4
    batch = 2

    rope_batched = Onion.STRING(dim, d_coords, 1)
    # Ensure STRINGBackup sees the same parameters as the batched variant
    rope_nonbatched = STRINGBackup.STRING(
        dim,
        d_coords,
        copy(rope_batched.thetas[:, 1]),
        copy(rope_batched.orthogonal_parameter[:, :, 1]),
    )

    for _ in 1:3
        positions = rand(Float32, d_coords, seq_len, batch)
        batched_out = dropdims(rope_batched(positions); dims=4)

        expected = Array{Float32}(undef, dim, dim, seq_len, batch)
        for b in 1:batch
            for s in 1:seq_len
                expected[:, :, s, b] = rope_nonbatched(positions[:, s, b])
            end
        end

        @test isapprox(batched_out, expected; rtol=1e-5, atol=1e-5)
    end
end
