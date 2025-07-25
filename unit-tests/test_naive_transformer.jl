using Pkg; Pkg.activate("..")

using Test
using Onion
using Flux
using Random; rng = Xoshiro(0)

@testset "NaiveTransformerBlock Forward Pass" begin
    # 1. Define Model Parameters
    dim = 48
    n_heads = 8
    d_coords = 3

    # 2. Instantiate the NaiveTransformerBlock
    # This creates a transformer block that uses the NaiveRoPE for positional embeddings.
    # The head dimension (dim / n_heads) must be divisible by 6.
    # Here, head_dim = 48 / 8 = 6, which is valid.
    transformer = NaiveTransformerBlock(dim, n_heads, d_coords)

    # 3. Create Sample Input Data
    seq_len = 10
    batch_size = 4
    
    # Create a random input tensor 'x' with the shape (dim, seq_len, batch_size)
    x = randn(rng, Float32, dim, seq_len, batch_size)
    
    # Create a random positions tensor with the shape (d_coords, seq_len, batch_size)
    positions = randn(rng, Float32, d_coords, seq_len, batch_size)

    # 4. Perform a Forward Pass
    # The input 'x' and the 'positions' are passed to the transformer block.
    # The NaiveRoPE inside the attention layer will use the positions to rotate the queries and keys.
    output = transformer(x, positions=positions)

    # 5. Verify the Output
    # The output of the transformer block should have the same shape as the input 'x'.
    @test size(output) == (dim, seq_len, batch_size)
    
    println("Successfully ran NaiveTransformerBlock forward pass.")
    println("Input shape: ", size(x))
    println("Output shape: ", size(output))
end 