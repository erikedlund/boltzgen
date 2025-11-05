# BoltzGen PPI Scoring Methodology

## Executive Summary

The current `boltzgen-affinity` script is **NOT suitable for protein-protein interaction (PPI) scoring**. It only works for protein-small molecule affinity prediction. For PPIs, BoltzGen uses a completely different methodology based on:

1. **Structure Quality Metrics** (from Boltz-2 folding)
2. **Interface Analysis** (hydrogen bonds, salt bridges, buried surface area)
3. **Rank-Based Aggregation** (multi-metric scoring)

This document explains how BoltzGen actually scores PPIs and outlines a plan to create a proper standalone PPI scoring tool.

---

## The Problem with Current `boltzgen-affinity`

The affinity prediction module (`AffinityModule` in `src/boltzgen/model/modules/affinity.py`) is **ONLY trained for protein-small molecule binding**. It:

- Takes small molecule (NONPOLYMER) entities as input
- Predicts binding affinity using a neural network trained on small molecule binding data
- **Does NOT work for protein-protein interactions**

When used for PPIs, it would either:
1. Fail (no NONPOLYMER entity found)
2. Give meaningless results (if forced to treat a protein chain as ligand)

---

## How BoltzGen Actually Scores PPIs

### Overview

BoltzGen evaluates protein-protein binders through a **multi-step pipeline**:

```
Design ’ Inverse Folding ’ Refolding ’ Analysis ’ Filtering/Ranking
```

The scoring happens in the **Analysis** and **Filtering** steps, which use structure-based metrics rather than a single affinity score.

### Step 1: Structure Refolding (Boltz-2)

After generating a binder design, BoltzGen refolds:
1. **The entire complex** (binder + target)
2. **The binder in isolation** (to check if it maintains its shape)

This validates that the design:
- Folds correctly with the target present
- Folds correctly without the target
- Maintains the same structure in both contexts

**Key Metrics from Folding:**
- `rmsd` / `bb_rmsd`: RMSD between designed and refolded structure
- `iptm`: Interface predicted TM-score (quality of interface contacts)
- `ptm`: Predicted TM-score (overall structure quality)
- `min_design_to_target_pae`: Minimum PAE (position accuracy error) at interface
- `lddt`: Local distance difference test

### Step 2: Interface Analysis

The Analyze task (`src/boltzgen/task/analyze/analyze.py`) computes detailed interface metrics:

#### Physical Interactions (computed using PLIP):
- `plip_hbonds`: Number of hydrogen bonds at interface
- `plip_saltbridge`: Number of salt bridges
- `plip_hydrophobic`: Number of hydrophobic contacts

#### Buried Surface Area:
- `delta_sasa_refolded`: Change in solvent-accessible surface area when binder binds
  - Computed by comparing binder+target complex vs binder alone
  - Larger values indicate more extensive interface

#### Hydrophobicity:
- `design_hydrophobicity`: Sequence-based hydrophobicity
- `design_largest_hydrophobic_patch_refolded`: Largest hydrophobic patch area

#### Sequence Composition:
- Amino acid fractions (e.g., `CYS_fraction`, `ALA_fraction`)
- Used to filter out biased sequences

#### Secondary Structure:
- `loop`, `helix`, `sheet`: Fractions of each secondary structure type

#### Design Metrics:
- `num_design`: Number of designed residues
- `designed_sequence`: The actual sequence
- `designed_chain_sequence`: Full chain sequence

### Step 3: Rank-Based Aggregation Scoring

The Filter task (`src/boltzgen/task/filter/filter.py`) implements a sophisticated **rank-based aggregation** algorithm:

#### Default Metrics for PPI Scoring (protein-anything protocol):

```python
metrics = {
    "design_to_target_iptm": 1,      # Interface quality (weight=1)
    "design_ptm": 1,                  # Binder structure quality (weight=1)
    "neg_min_design_to_target_pae": 1, # Interface confidence (weight=1)
    "plip_hbonds_refolded": 2,        # H-bonds (weight=2, less important)
    "plip_saltbridge_refolded": 2,    # Salt bridges (weight=2)
    "delta_sasa_refolded": 2,         # Buried surface (weight=2)
}
```

**Weight interpretation:** Higher weight = LESS important (rank is divided by weight)

#### The Ranking Algorithm:

```python
# For each design:
for metric, weight in metrics.items():
    # 1. Rank all designs by (num_filters_passed, metric)
    rank[metric] = rank_designs_by(num_filters_passed, metric)

    # 2. Scale rank by inverse importance weight
    scaled_rank[metric] = rank[metric] / weight

# 3. The worst (maximum) scaled rank becomes the quality score
quality_score = max(scaled_rank.values())

# 4. Designs are sorted by quality_score (lower is better)
final_rank = sort_by(quality_score, tiebreak_by=iptm)
```

**Key insight:** A design is only as good as its **worst metric**. If any single metric is poor, the design ranks low overall.

#### Hard Filters:

Before ranking, designs must pass mandatory thresholds:

```python
filters = [
    {"feature": "has_x", "threshold": 0},                    # No unknown residues
    {"feature": "filter_rmsd", "threshold": 2.5},            # Good refolding
    {"feature": "filter_rmsd_design", "threshold": 2.5},     # Design part refolds well
    {"feature": "designfolding-filter_rmsd", "threshold": 2.5}, # Isolated design refolds
    {"feature": "CYS_fraction", "threshold": 0},             # No designed cysteines (peptide protocol)
    {"feature": "ALA_fraction", "threshold": 0.2},           # Not alanine-rich
    {"feature": "GLY_fraction", "threshold": 0.2},           # Not glycine-rich
    # ... other composition filters
]
```

### Step 4: Diversity Selection

After ranking by quality, BoltzGen performs **diversity-aware selection**:

```python
# MaxMin lazy-greedy algorithm
def select_diverse_designs(k, quality_scores, sequences):
    selected = [argmax(quality_scores)]  # Start with best design

    for i in range(k - 1):
        # For each remaining design:
        for design in remaining:
            # Compute sequence identity to nearest selected design
            diversity = 1 - max(seq_identity(design, s) for s in selected)

            # Combined score: (1-alpha) * quality + alpha * diversity
            score = (1 - alpha) * quality[design] + alpha * diversity

        # Select design with best combined score
        selected.append(argmax(score))

    return selected
```

**Alpha parameter (default=0.1):**
- ±=0: Pure quality selection
- ±=1: Pure diversity selection
- ±=0.1: 90% quality, 10% diversity

---

## Proposed New Tool: `boltzgen-score-ppi`

### Architecture

```
Input: Complex structure (.cif)
   “
[1] Parse structure, identify chains
   “
[2] Refold complex using Boltz-2
   “
[3] Compute interface metrics (PLIP, SASA, etc.)
   “
[4] Apply ranking algorithm
   “
Output: Quality scores + metrics CSV
```

### Implementation Plan

#### Phase 1: Core Infrastructure
1. **Structure parsing and tokenization**
   - Reuse existing `mmcif.parse_mmcif()` and `Tokenizer`
   - No design masks needed (everything is "fixed")

2. **Refolding with Boltz-2**
   - Use `Predict` task with folding checkpoint
   - Extract confidence metrics (iPTM, PAE, pLDDT)

3. **Metric computation**
   - Port relevant functions from `analyze.py`:
     - `count_noncovalents()` for PLIP interactions
     - `get_delta_sasa()` for buried surface area
     - `compute_rmsd()` for structural comparison
   - Compute on both original and refolded structures

#### Phase 2: Ranking Implementation
1. **Implement rank-based aggregation**
   - Port ranking logic from `filter.py`
   - Make metrics configurable (allow user overrides)

2. **Hard filters**
   - Basic filters: RMSD, PAE thresholds
   - Composition filters (optional)

#### Phase 3: CLI Interface
```bash
boltzgen-score-ppi complex.cif \
  --checkpoint boltz2_fold.ckpt \
  --output ./results \
  --ligand-chains B,C \
  --receptor-chains A \
  --metrics design_to_target_iptm=1 plip_hbonds_refolded=2 \
  --rmsd-threshold 2.5
```

### Output Format

**CSV with columns:**
```
id, designed_sequence, num_design,
design_to_target_iptm, design_ptm, min_design_to_target_pae,
rmsd, bb_rmsd,
plip_hbonds_refolded, plip_saltbridge_refolded, plip_hydrophobic_refolded,
delta_sasa_refolded, design_sasa_bound_refolded, design_sasa_unbound_refolded,
rank_design_to_target_iptm, rank_design_ptm, rank_neg_min_design_to_target_pae,
max_rank, final_rank, quality_score
```

### Key Differences from Current Tool

| Feature | Current `boltzgen-affinity` | Proposed `boltzgen-score-ppi` |
|---------|----------------------------|-------------------------------|
| **Scope** | Protein-small molecule only | Protein-protein interactions |
| **Method** | Neural network affinity predictor | Structure-based metrics + ranking |
| **Output** | Single affinity score | Multiple metrics + composite rank |
| **Validation** | None | Refolding validation (RMSD, PAE) |
| **Interface metrics** | No | Yes (H-bonds, SASA, salt bridges) |

---

## Implementation Checklist

- [ ] Create new CLI script `src/boltzgen/cli/score_ppi.py`
- [ ] Implement structure preparation (similar to `predict_affinity.py`)
- [ ] Set up Boltz-2 refolding pipeline
- [ ] Port metric computation functions from `analyze.py`
  - [ ] PLIP interaction counting
  - [ ] SASA calculation
  - [ ] RMSD computation
- [ ] Implement rank-based aggregation from `filter.py`
- [ ] Add configurable metrics and weights
- [ ] Create output CSV writer
- [ ] Write tests with example PPI complexes
- [ ] Add to `pyproject.toml` entrypoints
- [ ] Update documentation

---

## Example Usage (Future)

```bash
# Score a single PPI complex
boltzgen-score-ppi 1a2k.cif \
  --checkpoint ~/.cache/boltz2_fold.ckpt \
  --output ./ppi_scores \
  --ligand-chains B \
  --receptor-chains A

# Score multiple complexes in a directory
boltzgen-score-ppi ./designs/ \
  --checkpoint ~/.cache/boltz2_fold.ckpt \
  --output ./batch_scores \
  --ligand-chains B \
  --devices 4

# Custom metric weights
boltzgen-score-ppi complex.cif \
  --checkpoint ~/.cache/boltz2_fold.ckpt \
  --output ./scores \
  --ligand-chains B \
  --metrics design_to_target_iptm=1 plip_hbonds_refolded=4 delta_sasa_refolded=3

# With hard filters
boltzgen-score-ppi complex.cif \
  --checkpoint ~/.cache/boltz2_fold.ckpt \
  --output ./scores \
  --ligand-chains B \
  --rmsd-threshold 3.0 \
  --min-iptm 0.6
```

---

## Technical Deep Dive: Rank-Based Aggregation

### Why Rank-Based Instead of Direct Scores?

The ranking algorithm uses **ranks** rather than raw metric values because:

1. **Different scales**: iptm  [0,1], SASA  [0, thousands], H-bonds  [0, ~20]
2. **Non-linear relationships**: A 0.1 difference in iptm at 0.9 is more meaningful than at 0.5
3. **Robustness**: Ranks are less sensitive to outliers
4. **Interpretability**: Rank=10 means "10th best" regardless of metric

### The Algorithm (from `filter.py:420-461`)

```python
# Step 1: Create rank dataframe
rank_df = pd.DataFrame(index=df.index)

# Step 2: For each metric, compute scaled rank
for col, inverse_importance in metrics.items():
    # Rank by (num_filters_passed, metric_value)
    # This pushes designs that fail filters to the bottom
    rank_df[f"rank_{col}"] = (
        df[["num_filters_passed", col]]
        .apply(tuple, axis=1)
        .rank(method="min", ascending=False)  # Higher is better
        .astype(int)
        / inverse_importance  # Scale by importance
    )

# Step 3: Take worst (max) scaled rank as quality key
df["max_rank"] = rank_df.max(axis=1)

# Step 4: Sort by max_rank, create dense final rank
df = df.sort_values("max_rank")
df["secondary_rank"] = df["max_rank"].rank(method="dense").astype(int)

# Step 5: Tiebreak by iptm
df = df.sort_values(
    by=["secondary_rank", "design_to_target_iptm"],
    ascending=[True, False]  # Lower rank, higher iptm
)

# Step 6: Assign final ranks
df["final_rank"] = np.arange(1, len(df) + 1)

# Step 7: Convert to quality score [0, 1]
df["quality_score"] = 1 - (df["final_rank"] - 1) / (len(df) - 1)
```

### Example Walkthrough

Suppose we have 3 designs and 3 metrics:

**Raw metrics:**
```
Design | iptm | PAE  | H-bonds | Filters Passed
-------|------|------|---------|---------------
A      | 0.9  | 5.0  | 8       | 3/3
B      | 0.85 | 4.5  | 12      | 3/3
C      | 0.95 | 8.0  | 6       | 2/3 (failed one)
```

**Metrics config:**
```python
metrics = {
    "iptm": 1,           # High importance
    "neg_pae": 1,        # High importance (negated, so lower PAE is better)
    "h_bonds": 2,        # Lower importance
}
```

**Step 1: Rank by (num_filters, metric)**

For iptm (higher is better):
```
Design | (filters, iptm) | Rank
-------|-----------------|------
C      | (2, 0.95)       | 3 (worst, only 2 filters)
A      | (3, 0.9)        | 2
B      | (3, 0.85)       | 1 (lowest of the 3/3 group)
```

For neg_pae (higher neg_pae = lower PAE = better):
```
Design | (filters, neg_pae) | Rank
-------|-------------------|------
C      | (2, -8.0)         | 3 (worst)
B      | (3, -4.5)         | 2 (best of 3/3 group)
A      | (3, -5.0)         | 1
```

For h_bonds (higher is better):
```
Design | (filters, h_bonds) | Rank
-------|-------------------|------
C      | (2, 6)            | 3
A      | (3, 8)            | 2
B      | (3, 12)           | 1 (best)
```

**Step 2: Scale ranks by importance**
```
Design | rank_iptm/1 | rank_neg_pae/1 | rank_h_bonds/2 | max_rank
-------|-------------|----------------|----------------|----------
A      | 2           | 1              | 1              | 2
B      | 1           | 2              | 0.5            | 2
C      | 3           | 3              | 1.5            | 3
```

**Step 3: Sort by max_rank ’ Final ranking:**
```
Final Rank | Design | max_rank | iptm | PAE | H-bonds
-----------|--------|----------|------|-----|--------
1          | B      | 2        | 0.85 | 4.5 | 12
2          | A      | 2        | 0.9  | 5.0 | 8
3          | C      | 3        | 0.95 | 8.0 | 6
```

**Key observations:**
- Design C ranks last despite best iptm, because it failed a filter
- A and B tie on max_rank, so tiebreaker (iptm) decides: B wins
- B's superior H-bonds don't help much (weight=2), but it's balanced overall

---

## References

- **Filter algorithm:** `src/boltzgen/task/filter/filter.py` (lines 420-461)
- **Analyze metrics:** `src/boltzgen/task/analyze/analyze.py`
- **Protocol configs:** `src/boltzgen/cli/boltzgen.py` (lines 76-110)
- **PLIP interactions:** `src/boltzgen/task/analyze/analyze_utils.py`
- **SASA calculation:** Uses FreeSASA library
- **Affinity module (small molecule only):** `src/boltzgen/model/modules/affinity.py`

---

## Conclusion

To create a proper PPI scoring tool for BoltzGen, we need to:

1. **NOT use the affinity module** (it's for small molecules)
2. **Implement the full analysis pipeline**: refolding ’ metrics ’ ranking
3. **Use rank-based aggregation** with multiple structure-quality metrics
4. **Make it configurable** so users can adjust metric weights and filters

This will provide a scientifically sound way to evaluate protein-protein complexes that matches how BoltzGen internally evaluates binder designs.
