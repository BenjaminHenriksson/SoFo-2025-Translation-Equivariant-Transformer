using Pkg; Pkg.activate("..")



# --- Load packages ---
#Pkg.add(["ProteinChains", "BSON", "Flux", "ConcreteStructs", "ChainRulesCore", "Einops", "RandomFeatureMaps"])
using Onion, ProteinChains, BSON, Flux, ConcreteStructs, ChainRulesCore, Einops, RandomFeatureMaps
using ForwardBackward, Flowfusion, Optimisers, Plots
using OneHotArrays


# --- Load data ---
BSON.@load "./pdb500_allatom.bson" allatom_dataset;
# length(allatom_dataset) == 500

# All different types of atoms (orderd tuple per amino acid)
ATOM_TYPES = String["C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3", "CG", "CG1", "CG2", "CH2", "CZ", "CZ2", "CZ3", "H", "H1", "H2", "H3", "HA", "HA2", "HA3", "HB", "HB1", "HB2", "HB3", "HD1", "HD11", "HD12", "HD13", "HD2", "HD21", "HD22", "HD23", "HD3", "HE", "HE1", "HE2", "HE21", "HE22", "HE3", "HG", "HG1", "HG11", "HG12", "HG13", "HG2", "HG21", "HG22", "HG23", "HG3", "HH", "HH11", "HH12", "HH2", "HH21", "HH22", "HZ", "HZ1", "HZ2", "HZ3", "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "OXT", "SD", "SG"]
AA_TO_ATOMS = Dict("Q" => ["C", "CA", "CB", "CD", "CG", "N", "NE2", "O", "OE1"], "W" => ["C", "CA", "CB", "CD1", "CD2", "CE2", "CE3", "CG", "CH2", "CZ2", "CZ3", "N", "NE1", "O"], "T" => ["C", "CA", "CB", "CG2", "N", "O", "OG1"], "P" => ["C", "CA", "CB", "CD", "CG", "N", "O"], "C" => ["C", "CA", "CB", "N", "O", "SG"], "V" => ["C", "CA", "CB", "CG1", "CG2", "N", "O"], "L" => ["C", "CA", "CB", "CD1", "CD2", "CG", "N", "O"], "M" => ["C", "CA", "CB", "CE", "CG", "N", "O", "SD"], "N" => ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"], "H" => ["C", "CA", "CB", "CD2", "CE1", "CG", "N", "ND1", "NE2", "O"], "A" => ["C", "CA", "CB", "N", "O"], "X" => ["C", "CA", "CB", "N", "O"], "D" => ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"], "G" => ["C", "CA", "N", "O"], "E" => ["C", "CA", "CB", "CD", "CG", "N", "O", "OE1", "OE2"], "Y" => ["C", "CA", "CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ", "N", "O", "OH"], "I" => ["C", "CA", "CB", "CD1", "CG1", "CG2", "N", "O"], "S" => ["C", "CA", "CB", "N", "O", "OG"], "K" => ["C", "CA", "CB", "CD", "CE", "CG", "N", "NZ", "O"], "R" => ["C", "CA", "CB", "CD", "CG", "CZ", "N", "NE", "NH1", "NH2", "O"], "F" => ["C", "CA", "CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ", "N", "O"])
    


# --- Define model ---
struct SCFlowModel{L}
    layers::L
end

Flux.@layer SCFlowModel

# Initialization
function SCFlowModel(embed_dim::Int, num_layers::Int, num_heads::Int)
    layers = (;
        # Add masked attention heads

        # -------- Layers for both start --------
        embed_time = Chain(RandomFourierFeatures(1 => embed_dim, 1f0), Dense(embed_dim => embed_dim)),

        AA_type_encoder = Dense(21 => embed_dim, 1f0, bias = false),
        AA_atom_encoder = Dense(21 => embed_dim, 1f0, bias = false),
        AA_idx_encoder = RandomFourierFeatures(1 => embed_dim, 1f0),
        
        cross_attention_layers = [Attention(embed_dim, num_heads) for _ in 1:num_layers],
        # -------- Layers for both end --------


        # -------- Atom stack start --------
        atom_type_encoder = Dense(length(ATOM_TYPES) => embed_dim, 1f0, bias = false),
        
        atom_transformers = [NaiveTransformerBlock(embed_dim, num_heads, 3) for _ in 1:num_layers],
        # -------- Atom stack end --------


        # -------- Backbone stack start --------
        backbone_transformers = [NaiveTransformerBlock(embed_dim, num_heads, 3) for _ in 1:num_layers],
        # -------- Backbone stack end --------

        # Output atom movement / velocity
        output_layer = Dense(embed_dim => 3),
    )

    return SCFlowModel(layers)
end

(m::SCFlowModel)(prepped_batch) = m(prepped_batch.AAs_res, prepped_batch.AAs_atom, prepped_batch.AA_idx_res, prepped_batch.AA_idx_atom, prepped_batch.backbone_distanceincrement, prepped_batch.atom_calpha_distance, prepped_batch.atom_xyz, prepped_batch.backbone_xyz, prepped_batch.atom_name)

# Forward / backward pass
function (model::SCFlowModel)(AAs_res, AAs_atom, AA_idx_res, AA_idx_atom, backbone_distanceincrement, atom_calpha_distance, atom_xyz, backbone_xyz, atom_name)
    # AAs_res is one-hot (21, n_AAs)
    # AA_idx_res is the idx of the associated AA (1, n_AAs)
    # AAs_atom is one-hot (21, n_atoms)
    # AA_idx_atom is the idx of the associated AA (1, n_atoms)
    # backbone_xyz is (3, 4, n_AAs) containing position N, CA, C, O
    # atom_xyz is (3, n_atoms)
    # atom_name is atom type (e.g. "C1"), (n_atom_types, n_atoms) one-hot encoded #N/CA/C/O need to be masked, backbone does not move
    l = model.layers

    # Residue embeddings
    @show size(AAs_res)
    x_res = l.AA_type_encoder(AAs_res) + l.AA_idx_encoder(AA_idx_res)
    # ^ position goes in through encoder in transformer

    # Sidechain atom embeddings
    x_atom = l.atom_type_encoder(atom_name) + l.AA_idx_encoder(AA_idx_atom) + l.AA_atom_encoder(AAs_atom) 
    # ^ position goes in through encoder in transformer

    for i in 1:length(l.atom_transformers)
        # Pass through backbone transformer stack
        x_res = l.backbone_transformers[i](x_res, backbone_xyz)
        
        # Pass through atom transformer stack
        x_atom = l.atom_transformers[i](x_atom, atom_xyz)
        
        # Cross attention from backbone to atom stack
        x_atom = l.cross_attention_layers[i](x_atom, x_res)
    end

    # Output (move atom)
    y_atom = l.output_layer(x_atom)

    return y_atom
end


# --- Sampling functions ---

# Data format of a single protein entry in `allatom_dataset`:
# 6-element Vector{Tuple{Symbol, Tuple{Int64, Vararg{Int64}}}}:       
# (:chainids, (284,)): Chain id for each residue
# (:AAs, (284,)): Amino acid type for each residue
# (:backbone_xyz, (3, 4, 284)): Coordinates for N, CA, C, O for each residue
# (:atom_xyz, (3, 4621)): Coordinates for all side chain atoms.
# (:atom_res, (4621,)): Which residue each atom associated with is.
# (:atom_name, (4621,)): atom_name ≈ atom type (e.g. "C1" = first carbon in side chain, etc).

"""
    sample_batch(dataset, n_samples)

Randomly samples `n_samples` proteins from the dataset.
"""
function sample_batch(dataset, n_samples)
    indices = rand(1:length(dataset), n_samples)
    return dataset[indices]
end

"""
    sampleX1(batch)

Get the ground truth side chain atom positions for a batch of proteins.
Returns a list of matrices of 3D coordinates.
"""
function sampleX1(batch)
    return [p.atom_xyz for p in batch]
end

"""
    sampleX0(batch; sigma=1.0f0)

Generates initial noisy positions for all side chain atoms of proteins in a batch.
Returns a list of matrices of noisy 3D coordinates.
"""
# Backbones are in all atom data but should not be moved
function sampleX0(batch; sigma=1.0f0)
    noisy_batch = []
    for protein in batch
        atom_xyz = protein.atom_xyz
        atom_res = protein.atom_res
        backbone_xyz = protein.backbone_xyz
        
        num_residues = size(backbone_xyz, 3)
        noisy_atom_xyz = similar(atom_xyz)

        for residue_idx in 1:num_residues
            atom_indices = findall(==(residue_idx), atom_res)
            num_atoms = length(atom_indices)

            if num_atoms == 0
                continue
            end

            ca_coord = backbone_xyz[:, 2, residue_idx]
            noise = randn(Float32, 3, num_atoms) .* sigma
            noisy_coords = ca_coord .+ noise
            noisy_atom_xyz[:, atom_indices] = noisy_coords
        end
        push!(noisy_batch, noisy_atom_xyz)
    end
    return noisy_batch
end

# --- Sampling functions end ----

# --- Data preprocessing ---

calc_dist(p1, p2) = Float32(sqrt(sum((p1 .- p2).^2)))

function prep_data(protein)
    maskcalpha = protein.atom_name .!= "CA"

    atom_xyz = Float32.(protein.atom_xyz[:, maskcalpha])
    atom_res = protein.atom_res[maskcalpha]

    atom_name = (onehotbatch(protein.atom_name[maskcalpha], ATOM_TYPES))
    AAs_res = (onehotbatch(protein.AAs, ProteinChains.AMINOACIDS))
    AAs_atom = (onehotbatch(protein.AAs[atom_res], ProteinChains.AMINOACIDS))
    
    n_res = length(protein.AAs)
    AA_idx_res = Float32.(reshape(1:n_res, 1, n_res))
    AA_idx_atom = Float32.(reshape(atom_res, 1, length(atom_res)))

    backbone_xyz = Float32.(protein.backbone_xyz[:, 2, :])

    backbone_distanceincrement = map(calc_dist, eachcol(backbone_xyz[:, 2:end]), eachcol(backbone_xyz[:, 1:end-1]))
    backbone_distanceincrement = [backbone_distanceincrement; zero(Float32)]
    backbone_distanceincrement = reshape(backbone_distanceincrement, 1, size(backbone_distanceincrement)...)

    atom_calpha_distance = map(calc_dist, eachcol(atom_xyz), eachcol(backbone_xyz[:, atom_res]))
    atom_calpha_distance = reshape(atom_calpha_distance, 1, size(atom_calpha_distance)...)

    return (; AAs_res, AAs_atom, AA_idx_res, AA_idx_atom, backbone_distanceincrement, atom_calpha_distance, atom_xyz, backbone_xyz, atom_name)
end

# --- Data preprocessing end ---















# --- Training process ---
# Embedding dimension, number of layers, number of heads
model = SCFlowModel(192, 4, 8) # ⇒ head_dim = 192 ÷ 8 = 24

# Flow process type
P = BrownianMotion(0.15f0) 
#P = Deterministic()

# Optimizer
eta = 0.001
opt_state = Flux.setup(AdamW(eta = eta), model)

# Training loop
iters = 4000
batch_size = 4
T = Float32
#=
for i in 1:iters
    # Set up a batch of training pairs, and t
    batch = sample_batch(allatom_dataset, batch_size)
    X0_data = sampleX0(batch)
    X1_data = sampleX1(batch)

    prepped_batch = prep_data.(batch)
    
    # Gradient & update
    loss, grad = Flux.withgradient(model) do m
        total_loss = 0.0f0
        for (p_data, x0, x1) in zip(prepped_batch, X0_data, X1_data)
            n_atoms = size(x1, 2)
            t = rand(T, 1, n_atoms) # time must be broadcastable to the state

            X0 = ContinuousState(x0)
            X1 = ContinuousState(x1)
            
            # Construct the bridge
            Xt = bridge(P, X0, X1, t)
            
            v = m(p_data)

            total_loss += floss(P, v, X1, scalefloss(P, t))
        end
        return total_loss / batch_size
    end
    Flux.update!(opt_state, model, grad[1])

    (i % 10 == 0) && println("i: $i; Loss: $loss")
end=#

batch_size = 4
batch = sample_batch(allatom_dataset, batch_size)
X0_data = vcat(sampleX0(batch)...)
X1_data = vcat(sampleX1(batch)...)

prepped_batch = prep_data.(batch)

X0 = ContinuousState(X0_data)
X1 = ContinuousState(X1_data)

t = rand(T, 1, size(X1.state, 2))

Xt = bridge(P, X0, X1, t)

loss, grad = Flux.withgradient(model) do m
    v = m(prepped_batch)
    return floss(P, v, X1, scalefloss(P, t))
end

Flux.update!(opt_state, model, grad[1])




# --- Template for generating side chain atoms ---
function makeX1predictor(data)
    b = prep_data(data)
    function f(X0)
        X1 = model(b.AAs_res, b.AAs_atom, b.AA_idx_res, b.AA_idx_atom, b.backbone_distanceincrement, b.atom_calpha_distance, X0, b.backbone_xyz, b.atom_name)
        return X1
    end
 
    return f
end
 #gen(P, X0, makeX1predictor(input))
 