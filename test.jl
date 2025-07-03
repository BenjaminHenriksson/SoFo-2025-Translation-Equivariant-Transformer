using Pkg; Pkg.activate(".")

# Make sure the Dev version of Onion is loaded (should print "dev env" when loading)
using Flux, DLProteinFormats, Onion, RandomFeatureMaps, StatsBase, Plots
using Test

dat = DLProteinFormats.load(PDBSimpleFlat500);

L = 30
train_inds = findall(dat.len .> L)
 
function random_batch(dat, L, B, filt_inds)
    locs = zeros(Float32, 3, L, B)
    inds = sample(filt_inds, B, replace=false)
    AAs = zeros(Int, L, B)
    for (i,ind) in enumerate(inds)
        l_range = rand(1:dat[ind].len - L + 1)
        locs[:, :, i] = dat[ind].locs[:, 1, l_range:l_range+L-1]
        AAs[:, i] = dat[ind].AAs[l_range:l_range+L-1]
    end
    return (;locs, AAs = Flux.onehotbatch(AAs, 1:20))
end

batch = random_batch(dat, L, 10, train_inds);
 
struct Toy1{L}
    layers::L
end
Flux.@layer Toy1
function Toy1(dim, depth)
    layers = (;
        loc_encoder = Dense(3 => dim, bias=false),
        transformers = [Onion.TransformerBlock(dim, 8, rope=Onion.MultiDimRoPE( Int(dim / 8), 3)) for _ in 1:depth],
        AA_decoder = Dense(dim => 20, bias=false),
    )
    return Toy1(layers)
end
function (m::Toy1)(locs)
    l = m.layers
    x = l.loc_encoder(locs)
    for transformerblock in l.transformers
        x = transformerblock(x, 0, nothing, x_pos = locs)
    end
    aa_logits = l.AA_decoder(x)
    return aa_logits
end

model = Toy1(64, 4)
opt_state = Flux.setup(AdamW(eta = 0.001), model)
losses = Float32[]

#=
for epoch in 1:20 # 1:100
    tot_loss = 0f0
    for i in 1:1_000 # 1:10_000
        batch = random_batch(dat, L, 10, train_inds)
        l, grad = Flux.withgradient(model) do m
            aalogits = m(batch.locs)
            Flux.logitcrossentropy(aalogits, batch.AAs)
        end
        Flux.update!(opt_state, model, grad[1])
        tot_loss += l
        if mod(i, 50) == 0
            println(epoch, " ", i, " ", tot_loss/50)
            push!(losses, tot_loss/50)
            tot_loss = 0f0
        end
        (mod(i, 500) == 0) && savefig(plot(losses), "losses_toy_MultiDimRoPE.pdf")
    end
end

# 42 minute runtime
=#

# Translation invariance tests for MultiDimRoPE
@testset "MultiDimRoPE Translation Invariance" begin
    dim = 32
    n_heads = 4
    seqlen = 6
    batch_size = 2

    attn = Onion.Attention(dim, n_heads; qkv_bias=false)
    rope = Onion.MultiDimRoPE(Int(dim / n_heads), 3)

    x = rand(Float32, dim, seqlen, batch_size)
    pos = rand(Float32, 3, seqlen, batch_size)

    function qkv(attn, x, pos)
        q = attn.wq(x)
        k = attn.wk(x)
        v = attn.wv(x)
        q = reshape(q, (attn.head_dim, attn.n_heads, seqlen, batch_size))
        k = reshape(k, (attn.head_dim, attn.n_kv_heads, seqlen, batch_size))
        v = reshape(v, (attn.head_dim, attn.n_kv_heads, seqlen, batch_size))
        q = permutedims(q, (1,3,2,4))
        k = permutedims(k, (1,3,2,4))
        v = permutedims(v, (1,3,2,4))
        q = rope(q, pos)
        k = rope(k, pos)
        return q, k, v
    end

    q1, k1, v1 = qkv(attn, x, pos)

    shift = rand(Float32, 3, 1, 1)
    pos_shift = pos .+ shift

    q2, k2, v2 = qkv(attn, x, pos_shift)

    @test isapprox(q1, q2; atol=1e-5, rtol=1e-5)
    @test isapprox(k1, k2; atol=1e-5, rtol=1e-5)
    @test isapprox(v1, v2; atol=1e-5, rtol=1e-5)
end

# Diagnostics for where invariance fails
@testset "MultiDimRoPE Invariance Breakdown" begin
    dim = 32
    n_heads = 4
    seqlen = 4
    batch_size = 1

    rope = Onion.MultiDimRoPE(Int(dim / n_heads), 3)
    x = rand(Float32, Int(dim / n_heads), seqlen, n_heads, batch_size)
    pos = rand(Float32, 3, seqlen, batch_size)
    shift = rand(Float32, 3, 1, 1)
    pos_shift = pos .+ shift

    function rope_parts(rope, x, pos)
        batchdims = size(x)[2:end]
        x_flat = reshape(x, :, prod(batchdims))
        pos_dims = size(pos)
        if ndims(pos) == 3
            pos_r = reshape(pos, pos_dims[1], pos_dims[2], 1, pos_dims[3])
        elseif ndims(pos) == 4
            pos_r = pos
        else
            error("positions must have 3 or 4 dimensions")
        end
        if size(pos_r, 3) == 1 && batchdims[2] > 1
            pos_r = repeat(pos_r, 1, 1, batchdims[2], 1)
        else
            @assert size(pos_r, 3) == batchdims[2]
        end
        pos_flat = reshape(pos_r, size(pos_r, 1), prod(batchdims))
        R = Onion.batched_mul(rope.Thetas, pos_flat)
        R_cis = cis.(R)
        x_perm = Onion.pairflip(x_flat)
        P = exp((rope.FreeMatrix - rope.FreeMatrix') .* 0.5)
        P_adj = reshape(P', size(P)..., 1)
        P_adj = repeat(P_adj, 1, 1, prod(batchdims))
        x_rot = (x_flat .* real.(R_cis)) .+ (x_perm .* imag.(R_cis))
        out = Onion.batched_vec(P_adj, x_rot)
        return (; R, x_perm, x_rot, out)
    end

    base = rope_parts(rope, x, pos)
    shifted = rope_parts(rope, x, pos_shift)

    @test !isapprox(base.R, shifted.R; atol=1e-5, rtol=1e-5)
    @test isapprox(base.x_perm, shifted.x_perm; atol=1e-5, rtol=1e-5)
    @test !isapprox(base.x_rot, shifted.x_rot; atol=1e-5, rtol=1e-5)
    @test !isapprox(base.out, shifted.out; atol=1e-5, rtol=1e-5)
end
