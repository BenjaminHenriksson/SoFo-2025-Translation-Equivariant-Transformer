<<<<<<< HEAD
using Pkg; Pkg.activate(".")
using Flux, DLProteinFormats, Onion, RandomFeatureMaps, StatsBase, Plots
dat = DLProteinFormats.load(PDBSimpleFlat);
 
L = 30
train_inds = findall(dat.len .> L)
 
function random_batch(dat, L, B, filt_inds)
    locs = zeros(Float32, 3, L, B)
    loc_diffs = zeros(Float32, 3, L, B)
    inds = sample(filt_inds, B, replace=false)
    AAs = zeros(Int, L, B)
    for (i,ind) in enumerate(inds)
        l_range = rand(1:dat[ind].len - L + 1)
        locs[:, :, i] = dat[ind].locs[:, 1, l_range:l_range+L-1]
        loc_diffs[:, 2:end, i] = locs[:, 1:end-1, i] .- locs[:, 2:end, i]
        AAs[:, i] = dat[ind].AAs[l_range:l_range+L-1]
    end
    return (;locs, loc_diffs, AAs = Flux.onehotbatch(AAs, 1:20))
end
 
batch = random_batch(dat, L, 10, train_inds);
 
 
struct ToyNaiveDiffs{L}
    layers::L
end
Flux.@layer ToyNaiveDiffs
function ToyNaiveDiffs(dim, depth)
    layers = (;
        loc_encoder = Chain(RandomFourierFeatures(3 => dim, 1f0), Dense(dim => dim, bias=false)),
        transformers = [Onion.NaiveTransformerBlock(dim, 8, 3, rope_theta = 100f0) for _ in 1:depth],
        AA_decoder = Dense(dim => 20, bias=false),
    )
    return ToyNaiveDiffs(layers)
end
function (m::ToyNaiveDiffs)(loc_diffs, locs)
    l = m.layers
    x = l.loc_encoder(loc_diffs)
    for layer in l.transformers
        x = layer(x, locs)
    end
    aa_logits = l.AA_decoder(x)
    return aa_logits
end
 
 
struct ToyRegularDiffs{L}
    layers::L
end
Flux.@layer ToyRegularDiffs
function ToyRegularDiffs(dim, depth)
    layers = (;
        loc_encoder = Chain(RandomFourierFeatures(3 => dim, 1f0), Dense(dim => dim, bias=false)),
        transformers = [Onion.TransformerBlock(dim, 8) for _ in 1:depth],
        AA_decoder = Dense(dim => 20, bias=false),
    )
    return ToyRegularDiffs(layers)
end
function (m::ToyRegularDiffs)(loc_diffs, locs)
    l = m.layers
    x = l.loc_encoder(loc_diffs)
    for layer in l.transformers
        x = layer(x, 0, nothing)
    end
    aa_logits = l.AA_decoder(x)
    return aa_logits
end


struct ToySTRINGDiffs{L}
    layers::L
end
Flux.@layer ToySTRINGDiffs
function ToySTRINGDiffs(dim, depth)
    layers = (;
        loc_encoder = Chain(RandomFourierFeatures(3 => dim, 1f0), Dense(dim => dim, bias=false)),
        transformers = [Onion.STRINGTransformerBlock(dim, 8, 3) for _ in 1:depth],
        AA_decoder = Dense(dim => 20, bias=false),
    )
    return ToySTRINGDiffs(layers)
end
function (m::ToySTRINGDiffs)(loc_diffs, locs)
    l = m.layers
    x = l.loc_encoder(loc_diffs)
    for layer in l.transformers
        x = layer(x; positions=locs)
    end
    aa_logits = l.AA_decoder(x)
    return aa_logits
end
 
 
model = ToyNaiveDiffs(96, 8); losses_filename = "losses_toy_naive.pdf"
#model = ToyRegularDiffs(96, 8); losses_filename = "losses_toy_regular.pdf"
#model = ToySTRINGDiffs(96, 8); losses_filename = "losses_toy_string.pdf"
 
opt_state = Flux.setup(AdamW(eta = 0.001), model)
 
losses = Float32[]
for epoch in 1:4
    tot_loss = 0f0
    for i in 1:10_000
        batch = random_batch(dat, L, 10, train_inds);
        l, grad = Flux.withgradient(model) do m
            aalogits = m(batch.loc_diffs, batch.locs)
            Flux.logitcrossentropy(aalogits, batch.AAs)
        end
        Flux.update!(opt_state, model, grad[1])
        tot_loss += l
        if mod(i, 50) == 0
            println(epoch, " ", i, " ", tot_loss/50)
            push!(losses, tot_loss/50)
            tot_loss = 0f0
        end
        (mod(i, 500) == 0) && savefig(plot(losses), losses_filename)
    end
end
 
=======
using Pkg; Pkg.activate(".")
using Flux, DLProteinFormats, Onion, RandomFeatureMaps, StatsBase, Plots
dat = DLProteinFormats.load(PDBSimpleFlat);
 
L = 30
train_inds = findall(dat.len .> L)
 
function random_batch(dat, L, B, filt_inds)
    locs = zeros(Float32, 3, L, B)
    loc_diffs = zeros(Float32, 3, L, B)
    inds = sample(filt_inds, B, replace=false)
    AAs = zeros(Int, L, B)
    for (i,ind) in enumerate(inds)
        l_range = rand(1:dat[ind].len - L + 1)
        locs[:, :, i] = dat[ind].locs[:, 1, l_range:l_range+L-1]
        loc_diffs[:, 2:end, i] = locs[:, 1:end-1, i] .- locs[:, 2:end, i]
        AAs[:, i] = dat[ind].AAs[l_range:l_range+L-1]
    end
    return (;locs, loc_diffs, AAs = Flux.onehotbatch(AAs, 1:20))
end
 
batch = random_batch(dat, L, 10, train_inds);
 
 
struct ToyNaiveDiffs{L}
    layers::L
end
Flux.@layer ToyNaiveDiffs
function ToyNaiveDiffs(dim, depth)
    layers = (;
        loc_encoder = Chain(RandomFourierFeatures(3 => dim, 1f0), Dense(dim => dim, bias=false)),
        transformers = [Onion.NaiveTransformerBlock(dim, 8, 3, rope_theta = 100f0) for _ in 1:depth],
        AA_decoder = Dense(dim => 20, bias=false),
    )
    return ToyNaiveDiffs(layers)
end
function (m::ToyNaiveDiffs)(loc_diffs, locs)
    l = m.layers
    x = l.loc_encoder(loc_diffs)
    for layer in l.transformers
        x = layer(x, locs)
    end
    aa_logits = l.AA_decoder(x)
    return aa_logits
end
 
 
struct ToyRegularDiffs{L}
    layers::L
end
Flux.@layer ToyRegularDiffs
function ToyRegularDiffs(dim, depth)
    layers = (;
        loc_encoder = Chain(RandomFourierFeatures(3 => dim, 1f0), Dense(dim => dim, bias=false)),
        transformers = [Onion.TransformerBlock(dim, 8) for _ in 1:depth],
        AA_decoder = Dense(dim => 20, bias=false),
    )
    return ToyRegularDiffs(layers)
end
function (m::ToyRegularDiffs)(loc_diffs, locs)
    l = m.layers
    x = l.loc_encoder(loc_diffs)
    for layer in l.transformers
        x = layer(x, 0, nothing)
    end
    aa_logits = l.AA_decoder(x)
    return aa_logits
end


struct ToySTRINGDiffs{L}
    layers::L
end
Flux.@layer ToySTRINGDiffs
function ToySTRINGDiffs(dim, depth)
    layers = (;
        loc_encoder = Chain(RandomFourierFeatures(3 => dim, 1f0), Dense(dim => dim, bias=false)),
        transformers = [Onion.STRINGTransformerBlock(dim, 8, 3) for _ in 1:depth],
        AA_decoder = Dense(dim => 20, bias=false),
    )
    return ToySTRINGDiffs(layers)
end
function (m::ToySTRINGDiffs)(loc_diffs, locs)
    l = m.layers
    x = l.loc_encoder(loc_diffs)
    for layer in l.transformers
        x = layer(x; positions=locs)
    end
    aa_logits = l.AA_decoder(x)
    return aa_logits
end
 
 
model = ToyNaiveDiffs(96, 8); losses_filename = "losses_toy_naive.pdf"
#model = ToyRegularDiffs(96, 8); losses_filename = "losses_toy_regular.pdf"
#model = ToySTRINGDiffs(96, 8); losses_filename = "losses_toy_string.pdf"
 
opt_state = Flux.setup(AdamW(eta = 0.001), model)
 
losses = Float32[]
for epoch in 1:4
    tot_loss = 0f0
    for i in 1:10_000
        batch = random_batch(dat, L, 10, train_inds);
        l, grad = Flux.withgradient(model) do m
            aalogits = m(batch.loc_diffs, batch.locs)
            Flux.logitcrossentropy(aalogits, batch.AAs)
        end
        Flux.update!(opt_state, model, grad[1])
        tot_loss += l
        if mod(i, 50) == 0
            println(epoch, " ", i, " ", tot_loss/50)
            push!(losses, tot_loss/50)
            tot_loss = 0f0
        end
        (mod(i, 500) == 0) && savefig(plot(losses), losses_filename)
    end
end
 
>>>>>>> 8307f037cf5c424ccfb61ea00a2125fc38d7ceda
