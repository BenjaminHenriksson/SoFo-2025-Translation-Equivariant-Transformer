using Pkg; Pkg.activate(".")

# Make sure the Dev version of Onion is loaded (should print "dev env" when loading)
using Flux, DLProteinFormats, Onion, RandomFeatureMaps, StatsBase, Plots, Statistics
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
 
function batched_pairwise_dist(x)
    sqnorms = sum(abs2, x, dims=1)
    A_sqnorms = reshape(sqnorms, size(x, 2), 1, size(x, 3))
    B_sqnorms = reshape(sqnorms, 1, size(x, 2), size(x, 3))
    return A_sqnorms .- 2 * (batched_transpose(x) ⊠ x) .+ B_sqnorms
end
 
#To handle singleton second dim:
batched_pairwise_dist(x::AbstractArray{T,4}) where T = batched_pairwise_dist(reshape(x, 3, size(x, 3), size(x, 4)))


# --- util -------------------------------------------------------------

"""
    pairwise_rff(locs, rff) -> Φ

Compute batched pair–wise **distances** (not squared),
apply the given `RandomFourierFeatures` layer,
and return an array of shape

    (64 , N , N , B)   # 64 = rff.out, N = number of residues, B = batch
"""
function pairwise_rff(locs, rff)
    D = batched_pairwise_dist(locs)            # (N , N , B), squared
    #D  = sqrt.(D .+ 1f-8)                      # distances, avoid sqrt(0)
    Dvec = reshape(D, 1, :, size(D,3))          # (1 , N*N , B)
    Φvec = rff(Dvec)                            # (64, N*N,  B)
    reshape(Φvec, 64, size(D,1), size(D,2), :)  # (64, N, N, B)
end

# --- model ------------------------------------------------------------

struct ToyTI{L}         # “TI” = translationally invariant
    layers::L
end
Flux.@layer ToyTI

function ToyTI(dim::Int, depth::Int)
    layers = (;                        # keep names close to the old ones
        dist_rff    = RandomFourierFeatures(1 => 64, 0.1f0),
        dist_encode = Dense(64 => dim, bias = false),
        transformers = [Onion.STRINGTransformerBlock(dim, 8, 3)
                        for _ in 1:depth],
        aa_decode   = Dense(dim => 20, bias = true),
    )
    ToyTI(layers)
end

function (m::ToyTI)(locs)
    l = m.layers

    # --- translational-invariant geometry embedding ------------------
    Φ = pairwise_rff(locs, l.dist_rff)          # (64, N, N, B)
    x = dropdims(mean(Φ; dims = 3); dims = 3)   # average over j  → (64, N, B)

    x = l.dist_encode(x)                        # (dim, N, B)

    # --- sequence modelling -----------------------------------------
    for block in l.transformers
        x = block(x, positions = locs)
    end

    return l.aa_decode(x)                       # (20, N, B)  – amino-acid logits
end

model = ToyTI(64, 4)
opt_state = Flux.setup(AdamW(eta = 0.001), model)
losses = Float32[]

for epoch in 1:20 # 1:100
    tot_loss = 0f0
    for i in 1:1_000 # 1:10_000
        batch = random_batch(dat, L, 10, train_inds)
        l, grad = Flux.withgradient(model) do m
            random_shift = rand(Float32, 3)
            aalogits = m(batch.locs .+ random_shift)
            Flux.logitcrossentropy(aalogits, batch.AAs)
        end
        Flux.update!(opt_state, model, grad[1])
        tot_loss += l
        if mod(i, 50) == 0
            println(epoch, " ", i, " ", tot_loss/50)
            push!(losses, tot_loss/50)
            tot_loss = 0f0
        end
        (mod(i, 500) == 0) && savefig(plot(losses), "losses_toy_STRING_RFF.pdf")
    end
end
