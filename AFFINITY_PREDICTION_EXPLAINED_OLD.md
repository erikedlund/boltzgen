# Affinity Prediction in BoltzGen - Technical Deep Dive

This document explains how affinity prediction works in BoltzGen, from input preparation through model inference to final outputs.

## Overview

Affinity prediction in BoltzGen estimates the binding strength between two molecular partners (protein-protein, protein-ligand, etc.) based on their 3D structures. The method uses a neural network trained to predict binding affinity from structural and geometric features of the interface.

## Pipeline Flow

### 1. Input Preparation (`boltzgen-affinity` CLI)

**File:** `src/boltzgen/cli/predict_affinity.py`

When you run:
```bash
boltzgen-affinity complex.cif --ligand-chains B --checkpoint ckpt.ckpt --output ./results
```

#### Step 1a: Structure Parsing
```python
# Parse the structure file (.cif or .pdb)
structure = mmcif.parse_mmcif(structure_path, canonicals, moldir=moldir)
# OR
structure = parse_pdb(structure_path, moldir=moldir)
```

The `Structure` object contains:
- **chains**: List of chains with metadata (name, mol_type, sequences)
- **residues**: List of residues with coordinates, residue types
- **atoms**: Atomic coordinates and properties
- **bonds**: Covalent bonds between atoms/residues

#### Step 1b: Tokenization
```python
tokenizer = Tokenizer(atomize_modified_residues=False)
tokenized = tokenizer.tokenize(structure, inverse_fold=False)
```

**Tokenization** converts residues/atoms into "tokens" (the basic units the model operates on):
- Each standard residue becomes one token
- Small molecules may be split into multiple tokens or atomized
- Returns `tokenized.tokens` dict with:
  - `asym_id`: Chain ID for each token (integer)
  - `mol_type`: 0=PROTEIN, 1=DNA, 2=RNA, 3=NONPOLYMER
  - `res_type`: Residue/atom type (integer, will be one-hot encoded)
  - `frame_rot`, `frame_t`: Local coordinate frames
  - Many other features...

#### Step 1c: Metadata Generation

This is where we specify **which tokens are ligand vs receptor**:

```python
# For protein-protein: user specifies chains
chain_names = [structure.chains[cid]["name"] for cid in chain_ids]
affinity_token_mask = np.array([name in ligand_chains for name in chain_names], dtype=bool)

# For protein-small molecule: auto-detect NONPOLYMER
affinity_token_mask = mol_type == const.chain_type_ids["NONPOLYMER"]

# Save metadata (including the critical affinity_token_mask)
np.savez_compressed(
    output_path,
    design_mask=np.zeros(num_tokens),      # All zeros for evaluation
    mol_type=mol_type,                      # Token types from tokenization
    affinity_token_mask=affinity_token_mask,  # CRITICAL: which tokens are ligand
    # ... other fields
)
```

**Key insight**: The `affinity_token_mask` determines which tokens are treated as the "ligand" (binding partner) vs "receptor". The model computes affinity by analyzing the interface between these two groups.

---

### 2. Data Loading & Featurization

**File:** `src/boltzgen/task/predict/data_from_generated.py`

#### Step 2a: Load Structure and Metadata
```python
# FromGeneratedDataset.__getitem__()
metadata = np.load(metadata_path)
design_mask = metadata["design_mask"]
affinity_token_mask = metadata.get("affinity_token_mask")  # Our saved mask!

# Parse structure again (to get coordinates, atoms, etc.)
structure = mmcif.parse_mmcif(structure_path, ...)
tokenized = tokenizer.tokenize(structure)
```

#### Step 2b: Featurization
```python
# Featurizer.process() - in src/boltzgen/data/feature/featurizer.py
features = featurizer.process(
    input_data,
    molecules=molecules,
    training=False,
    compute_affinity=True,  # Important flag!
    ...
)
```

The featurizer computes:
- **Single (token) features** (`s`-type, shape `[N_tokens, D_s]`):
  - `res_type`: One-hot residue type
  - `residue_index`: Position in chain
  - `asym_id`, `entity_id`: Chain/entity IDs
  - `mol_type`: Molecule type
  - MSA features (if available)
  - Template features

- **Pair (token-token) features** (`z`-type, shape `[N_tokens, N_tokens, D_z]`):
  - Relative positions
  - Distance histograms (if templates available)
  - Relational encodings

- **Atom features** (shape `[N_atoms, ...]`):
  - `coords`: Atomic coordinates
  - `atom_pad_mask`: Which atoms are real vs padding
  - `atom_to_token`: Mapping atoms → tokens
  - `token_to_rep_atom`: Mapping tokens → representative atoms (e.g., Cα)

#### Step 2c: Set Affinity Mask
```python
# In get_feat() - line 506-513
if affinity_token_mask is not None:
    features["affinity_token_mask"] = torch.from_numpy(affinity_token_mask).bool()
else:
    # Fallback: auto-detect NONPOLYMER
    features["affinity_token_mask"] = (
        features["mol_type"] == const.chain_type_ids["NONPOLYMER"]
    )
```

Now `features` is a complete dict ready for the model!

---

### 3. Model Inference

**File:** `src/boltzgen/model/models/boltz.py`

#### Step 3a: Forward Pass Through Main Model
```python
# Boltz.predict_step() calls Boltz.forward()
out = self(
    features,
    recycling_steps=3,
    num_sampling_steps=200,
    diffusion_samples=5,
    ...
)
```

The main `Boltz.forward()` does:
1. **Input Embedding**: Embed single/pair features
   ```python
   s_inputs = self.input_embedder(feats)  # [B, N, D_s]
   z = self.rel_pos_encoder(feats)        # [B, N, N, D_z]
   ```

2. **Recycling Loop** (repeat R times, default R=3):
   ```python
   for _ in range(recycling_steps):
       # Pairformer: attention-based processing of pair features
       s, z = self.pairformer_module(s_inputs, z, feats)

       # Structure module: predict 3D coordinates
       coords = self.structure_module(s, z, feats, ...)

       # Update features for next cycle
       z = update_pair_features_with_coords(z, coords)
   ```

3. **Confidence Prediction** (if enabled):
   ```python
   confidence_out = self.confidence_module(
       s_inputs, s, z, coords, feats
   )
   # Returns: ptm, iptm, plddt, pae, etc.
   ```

4. **Diffusion Sampling** (if doing generative design):
   - Not used for affinity prediction on fixed structures
   - Skip this for now

At this point we have:
- `s`: Final single token representations `[B, N_tokens, D_s]`
- `z`: Final pair token representations `[B, N_tokens, N_tokens, D_z]`
- `coords`: Predicted/refined atomic coordinates

#### Step 3b: Affinity Prediction Module

**File:** `src/boltzgen/model/modules/affinity.py`

The affinity prediction happens in `Boltz.forward()` lines 752-850:

```python
if self.affinity_prediction:
    # 1. Create interface mask
    pad_token_mask = feats["token_pad_mask"][0]  # Valid tokens
    rec_mask = (feats["mol_type"][0] == 0)       # Receptor (protein)
    lig_mask = feats["affinity_token_mask"][0]   # Ligand (user-specified!)

    # Cross-interface pairs: lig-rec, rec-lig, lig-lig
    cross_pair_mask = (
        lig_mask[:, None] * rec_mask[None, :]  # lig × rec
        + rec_mask[:, None] * lig_mask[None, :]  # rec × lig
        + lig_mask[:, None] * lig_mask[None, :]  # lig × lig
    )

    # 2. Mask pair features to interface only
    z_affinity = z * cross_pair_mask[None, :, :, None]

    # 3. Select best sample (by iptm)
    best_idx = torch.argmax(dict_out["iptm"])
    coords_affinity = dict_out["sample_atom_coords"][best_idx]

    # 4. Re-embed inputs for affinity
    s_inputs = self.input_embedder(feats, affinity=True)

    # 5. Call affinity module (possibly ensemble of 2 models)
    if self.affinity_ensemble:
        aff_out1 = self.affinity_module1(s_inputs, z_affinity, coords_affinity, feats)
        aff_out2 = self.affinity_module2(s_inputs, z_affinity, coords_affinity, feats)

        # Average predictions
        dict_out["affinity_pred_value"] = (
            aff_out1["affinity_pred_value"] + aff_out2["affinity_pred_value"]
        ) / 2
        dict_out["affinity_probability_binary"] = (
            sigmoid(aff_out1["affinity_logits_binary"]) +
            sigmoid(aff_out2["affinity_logits_binary"])
        ) / 2
    else:
        aff_out = self.affinity_module(s_inputs, z_affinity, coords_affinity, feats)
        dict_out.update(aff_out)
```

#### Step 3c: Inside AffinityModule

**File:** `src/boltzgen/model/modules/affinity.py:76-138`

```python
def forward(self, s_inputs, z, x_pred, feats, multiplicity=1, use_kernels=False):
    # 1. Update pair features with single features
    z = self.z_linear(self.z_norm(z))
    z = z + self.s_to_z_prod_in1(s_inputs)[:, :, None, :] + \
            self.s_to_z_prod_in2(s_inputs)[:, None, :, :]

    # 2. Compute distance features from predicted coordinates
    x_pred_repr = torch.bmm(feats["token_to_rep_atom"], x_pred)  # Token coords
    d = torch.cdist(x_pred_repr, x_pred_repr)  # Pairwise distances

    # 3. Discretize distances into bins
    distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1)
    distogram_embed = self.dist_bin_pairwise_embed(distogram)

    # 4. Add distance features to pair representation
    z = z + self.pairwise_conditioner(z_trunk=z, token_rel_pos_feats=distogram_embed)

    # 5. Apply Pairformer to interface pairs only
    rec_mask = (feats["mol_type"] == 0)
    lig_mask = feats["affinity_token_mask"]
    cross_pair_mask = (
        lig_mask[:, :, None] * rec_mask[:, None, :] +
        rec_mask[:, :, None] * lig_mask[:, None, :] +
        lig_mask[:, :, None] * lig_mask[:, None, :]
    )

    z = self.pairformer_stack(z, pair_mask=cross_pair_mask, use_kernels=use_kernels)

    # 6. Predict affinity from interface features
    out_dict = self.affinity_heads(z, feats, multiplicity)

    return out_dict
```

#### Step 3d: Affinity Heads

**File:** `src/boltzgen/model/modules/affinity.py:177-222`

```python
def forward(self, z, feats, multiplicity=1):
    # 1. Recreate interface mask
    rec_mask = (feats["mol_type"] == 0).unsqueeze(-1)
    lig_mask = feats["affinity_token_mask"].unsqueeze(-1)
    cross_pair_mask = (
        lig_mask[:, :, None] * rec_mask[:, None, :] +
        rec_mask[:, :, None] * lig_mask[:, None, :] +
        lig_mask[:, :, None] * lig_mask[:, None, :]
    ) * (1 - torch.eye(N_tokens))  # Exclude self-pairs

    # 2. Global pooling: average interface pair features
    g = torch.sum(z * cross_pair_mask, dim=(1, 2)) / torch.sum(cross_pair_mask, dim=(1, 2))
    # g shape: [B, D_z] - single vector representing the interface

    # 3. MLP to produce affinity embedding
    g = self.affinity_out_mlp(g)  # [B, D_hidden]

    # 4. Two prediction heads
    affinity_pred_value = self.to_affinity_pred_value(g)  # Continuous value (pK_d or similar)
    affinity_pred_score = self.to_affinity_pred_score(g)  # Score for binary classification
    affinity_logits_binary = self.to_affinity_logits_binary(affinity_pred_score)

    return {
        "affinity_pred_value": affinity_pred_value,      # Main output: predicted binding strength
        "affinity_logits_binary": affinity_logits_binary,  # Binary: binds or not
    }
```

**Key operations**:
1. **Global pooling**: Average all interface pair representations → single vector
2. **MLP processing**: Transform interface vector through MLPs
3. **Two heads**:
   - `affinity_pred_value`: Regression output (binding affinity, e.g., pKd)
   - `affinity_logits_binary`: Classification output (will bind / won't bind)

---

### 4. Output Writing

**File:** `src/boltzgen/task/predict/writer.py:110-157` (AffinityWriter)

```python
def write_on_batch_end(self, prediction, batch, ...):
    # Extract affinity predictions
    pred_dict = {}
    for key, value in prediction.items():
        if key in const.eval_keys:  # Includes affinity keys
            pred_dict[key] = value.cpu().numpy()

    # Save to NPZ file
    np.savez_compressed(
        self.outdir / f"{batch['id'][0]}.npz",
        **pred_dict
    )
```

The output `.npz` file contains:
```python
{
    "affinity_pred_value": array([[value]]),              # Main prediction
    "affinity_probability_binary": array([[prob]]),       # P(binds)
    # If ensemble:
    "affinity_pred_value1": ...,
    "affinity_pred_value2": ...,
    "affinity_probability_binary1": ...,
    "affinity_probability_binary2": ...,
    # Plus confidence metrics from earlier:
    "iptm": ...,
    "ptm": ...,
    "plddt": ...,
    # And structural info:
    "coords": ...,
    "res_type": ...,
    ...
}
```

---

## Summary: What Goes Into Affinity Prediction

### Inputs:
1. **Structure**: 3D coordinates of both binding partners
2. **Ligand/Receptor specification**: Which chains/residues are the "ligand" vs "receptor"
3. **Trained model**: Neural network weights trained on binding affinity data

### Processing:
1. **Tokenization**: Convert structure to tokens (residues/atoms)
2. **Feature computation**: MSA, templates, geometric features
3. **Pairformer**: Learn token representations through self-attention
4. **Structure refinement**: Optimize 3D coordinates (optional)
5. **Interface selection**: Mask to ligand-receptor pairs only
6. **Affinity-specific processing**:
   - Add distance features from 3D structure
   - Apply Pairformer to interface pairs
   - Global pooling of interface representation
   - MLP prediction heads

### Outputs:
1. **`affinity_pred_value`**: Predicted binding affinity (continuous value, likely pKd or ΔG)
2. **`affinity_probability_binary`**: Probability that the complex binds
3. **Confidence metrics**: iptm, ptm, plddt from structure prediction

---

## Key Architecture Choices

1. **Interface-focused**: Only pair features at the binding interface are used for affinity prediction. This is enforced by the `cross_pair_mask` which zeros out all non-interface pairs.

2. **Geometry-aware**: The distance features (`distogram`) from the 3D structure are explicitly added, allowing the model to consider geometric complementarity.

3. **Pairformer processing**: Additional Pairformer layers specifically for affinity allow the model to further refine interface representations.

4. **Global pooling**: The entire interface is summarized into a single vector by averaging all interface pair features. This is a strong inductive bias that binding affinity is a function of the overall interface, not local patches.

5. **Ensemble**: Using 2 independently trained affinity modules and averaging can improve robustness.

6. **Best structure selection**: When multiple structure samples exist (from diffusion), the one with highest iptm is chosen for affinity prediction.

---

## For Protein-Protein Affinity

The critical difference for protein-protein vs protein-ligand:
- **Protein-ligand**: `affinity_token_mask = (mol_type == NONPOLYMER)`
- **Protein-protein**: `affinity_token_mask = user_specified_chains`

Both receptor and ligand are proteins (mol_type=PROTEIN), so we **must** explicitly specify which chains are which via the CLI:
```bash
--ligand-chains B,C --receptor-chains A
```

This mask is saved in the metadata `.npz` file and used throughout the pipeline to identify the interface.
