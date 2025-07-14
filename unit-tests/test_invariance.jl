using Pkg; Pkg.activate(".")

# Make sure the Dev version of Onion is loaded (should print "dev env" when loading)
using Onion, Test, Flux

@testset "STRING Invariance" begin
    dim = 8
    d_coords = 3
    seq_len = 10
    batches = 5
    rope = Onion.STRING(dim, d_coords, 1)

    for _ in 1:3
        xi = rand(Float32, d_coords, seq_len, batches)
        xj = rand(Float32, d_coords, seq_len, batches)

        # lhs = rope(xi)' * rope(xj), but batched
        lhs = batched_mul(permutedims(dropdims(rope(xi); dims=4), (2, 1, 3, 4)), dropdims(rope(xj); dims=4))
        rhs = dropdims(rope(xj .- xi); dims=4)
        
        shift = rand(Float32, d_coords)
        shifted_rhs = batched_mul(permutedims(dropdims(rope(xi .+ shift); dims=4), (2, 1, 3, 4)), dropdims(rope(xj .+ shift); dims=4))

        # Tests basic invariance
        @test isapprox(lhs, rhs; rtol=1e-5, atol=1e-5)

        # Tests invariance after shift
        @test isapprox(lhs, shifted_rhs; rtol=1e-5, atol=1e-5)
        
        # Test to confirm actual variance after shift
        @test !isapprox(batched_mul(permutedims(dropdims(rope(xi .+ shift); dims=4), (2, 1, 3, 4)), dropdims(rope(xj); dims=4)), lhs)
        @test !isapprox(dropdims(rope(xj .- xi .+ shift); dims=4), rhs)
    end
end