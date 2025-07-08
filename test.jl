using Pkg; Pkg.activate(".")

# Make sure the Dev version of Onion is loaded (should print "dev env" when loading)
using Flux, Onion, Test, StatsBase

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

batch = random_batch(dat, L, 10, train_inds)
 