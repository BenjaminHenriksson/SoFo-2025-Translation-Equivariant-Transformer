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
        loc_encoder = Dense(0 => dim; bias = rand(dim)),
        transformers = [Onion.STRINGTransformerBlock(dim, 8, 3) for _ in 1:depth],
        AA_decoder = Dense(dim => 20, bias=true),
    )
    return Toy1(layers)
end
function (m::Toy1)(locs)
    l   = m.layers

    # 1.  Get the learnable constant token as a Float32 vector
    #     a) either call the 0→dim Dense with a typed empty vector …
    tok = l.loc_encoder(Float32[])          # length = dim, eltype = Float32
    #     b) … or simply take its bias (same numbers, a bit cleaner):
    # tok = l.loc_encoder.bias

    # 2.  Expand to (dim, L, B) without in-place mutation
    L, B = size(locs, 2), size(locs, 3)
    tok  = reshape(tok, :, 1, 1)                       # (dim,1,1)
    x    = tok .* ones(Float32, 1, L, B)               # (dim,L,B)

    # 3.  Usual forward pass
    for tr in l.transformers
        x = tr(x; positions = locs)
    end
    return l.AA_decoder(x)
end

model = Toy1(64, 4)
opt_state = Flux.setup(AdamW(eta = 0.001), model)
losses = Float32[]

#model_inital = deepcopy(model)

i = 1
epoch = 1
# for epoch in 1:20 # 1:100
    tot_loss = 0f0
#    for i in 1:1_000 # 1:10_000
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
        (mod(i, 500) == 0) && savefig(plot(losses), "total_invariance.pdf")
#    end
# end

#println(model_inital)
#println(model)