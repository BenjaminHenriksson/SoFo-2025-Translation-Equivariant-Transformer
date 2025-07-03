using Pkg; Pkg.activate(".")

# Make sure the Dev version of Onion is loaded (should print "dev env" when loading)
using Flux, DLProteinFormats, Onion, RandomFeatureMaps, StatsBase, Plots

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
