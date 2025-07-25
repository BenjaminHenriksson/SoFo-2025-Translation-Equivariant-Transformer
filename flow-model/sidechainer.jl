using Pkg; Pkg.activate("..")



# --- Load packages ---
#Pkg.add(["ProteinChains", "BSON", "Flux", "ConcreteStructs", "ChainRulesCore", "Einops", "RandomFeatureMaps"])
using Onion, ProteinChains, BSON, Flux, ConcreteStructs, ChainRulesCore, Einops, RandomFeatureMaps
using ForwardBackward, Flowfusion, Optimisers, Plots



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
        # -------- Layers for both start --------
        embed_time = Chain(RandomFourierFeatures(1 => embed_dim, 1f0), Dense(embed_dim => embed_dim)),

        AA_type_encoder = Dense(21 => embed_dim, 1f0, bias = false),
        AA_idx_encoder = RandomFourierFeatures(1 => embed_dim, 1f0),
        
        cross_attention_layers = [Attention(embed_dim, num_heads) for _ in 1:num_layers],
        # -------- Layers for both end --------


        # -------- Atom stack start --------
        atom_type_encoder = RandomFourierFeatures(length(ATOM_TYPES) => embed_dim, 1f0),
        
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

# Forward / backward pass
function (model::SCFlowModel)(AA_type, AA_idx, backbone_xyz, atom_xyz, atom_name)
    # AA_type is one-hot (21, n_AAs)
    # AA_idx is the idx of the associated AA (1, n_atoms)
    # backbone_xyz is (3, 4, n_AAs) containing position N, CA, C, O
    # atom_xyz is (3, n_atoms)
    # atom_name is atom type (e.g. "C1"), (n_atom_types, n_atoms) one-hot encoded #N/CA/C/O need to be masked, backbone does not move
    l = model.layers

    # Residue embeddings
    x_res = l.AA_type_encoder(AA_type) + l.AA_idx_encoder(AA_idx) 
    # ^ position goes in through encoder in transformer

    # Sidechain atom embeddings
    x_atom = l.atom_type_encoder(atom_name) + l.AA_idx_encoder(AA_idx) + l.AA_type_encoder(AA_type) 
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

for i in 1:iters
    # Set up a batch of training pairs, and t
    batch = sample_batch(allatom_dataset, batch_size)
    X0_data = sampleX0(batch)
    X1_data = sampleX1(batch)
    
    # Gradient & update
    loss, grad = Flux.withgradient(model) do m
        total_loss = 0.0f0
        for (protein, x0, x1) in zip(batch, X0_data, X1_data)
            n_atoms = size(x1, 2)
            t = rand(T, 1, n_atoms) # time must be broadcastable to the state

            X0 = ContinuousState(x0)
            X1 = ContinuousState(x1)
            
            # Construct the bridge
            Xt = bridge(P, X0, X1, t)
            
            v = m(
                protein.AAs, 
                protein.atom_res', # Check dimensions
                protein.backbone_xyz,
                tensor(Xt),
                protein.atom_name
            )

            total_loss += floss(P, v, X1, scalefloss(P, t))
        end
        return total_loss / batch_size
    end
    Flux.update!(opt_state, model, grad[1])

    (i % 10 == 0) && println("i: $i; Loss: $loss")
end


