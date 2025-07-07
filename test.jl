using Pkg; Pkg.activate(".")

# Make sure the Dev version of Onion is loaded (should print "dev env" when loading)
using Onion, Test

@testset "STRING Equivariance" begin
    dim = 8
    d_coords = 2
    rope = Onion.STRING(dim, d_coords)

    for _ in 1:3
        xi = randn(Float32, d_coords)
        xj = randn(Float32, d_coords)

        lhs = rope(xi)' * rope(xj)
        rhs = rope(xj .- xi)

        @test isapprox(lhs, rhs; rtol=1e-5, atol=1e-5)
    end
end