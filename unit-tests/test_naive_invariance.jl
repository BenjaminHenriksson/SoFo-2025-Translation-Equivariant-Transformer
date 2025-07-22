using Pkg; Pkg.activate("..")

using Test
using Onion
using Flux
using Random; rng = Xoshiro(0)

@testset "NaiveRoPE Translational Invariance" begin
    # 1. Define Model and Data Parameters
    dim = 48
    n_heads = 8
    d_coords = 3
    seq_len = 10
    batch_size = 4

    # 2. Instantiate the NaiveTransformerBlock
    transformer = NaiveTransformerBlock(dim, n_heads, d_coords)

    # 3. Create sample data
    x = randn(rng, Float32, dim, seq_len, batch_size)
    positions = randn(rng, Float32, d_coords, seq_len, batch_size)

    # 4. Create a random shift and apply it to the positions
    random_shift = randn(rng, Float32, d_coords, 1, 1)
    shifted_positions = positions .+ random_shift

    # 5. Perform forward passes with both original and shifted positions
    output_original = transformer(x, positions=positions)
    output_shifted = transformer(x, positions=shifted_positions)

    # 6. Check for invariance
    # The outputs should be nearly identical.
    @test output_original ≈ output_shifted

    # 7. Check for variance with a non-global shift
    # Create a random shift for each position in the sequence and batch.
    non_global_shift = randn(rng, Float32, d_coords, seq_len, batch_size)
    non_globally_shifted_positions = positions .+ non_global_shift
    
    # Perform a forward pass with the non-globally shifted positions.
    output_non_global_shifted = transformer(x, positions=non_globally_shifted_positions)
    
    # The output should now be different from the original.
    @test !(output_original ≈ output_non_global_shifted)
end 