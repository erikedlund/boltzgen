# BoltzGen PPI Scoring Tool

## Overview

`boltzgen-score-ppi` is a standalone tool for scoring protein-protein interactions (PPIs) using the same methodology as BoltzGen's internal pipeline, but without requiring structure refolding.

**Key Features:**
- Scores arbitrary PPI complexes using structure quality metrics
- Uses Boltz-2 confidence prediction (iPTM, PAE, pLDDT)
- Implements rank-based aggregation for multi-metric scoring
- No refolding required - works directly on input structures
- Batch processing support for multiple complexes

## Installation

The tool is included with BoltzGen. After installing the package:

```bash
pip install -e .
```

The `boltzgen-score-ppi` command will be available.

## Quick Start

### Score a single PPI complex

```bash
boltzgen-score-ppi complex.cif \
  --checkpoint ~/.cache/boltzgen/boltz2.ckpt \
  --output ./results \
  --ligand-chains B \
  --receptor-chains A
```

### Score multiple complexes in a directory

```bash
boltzgen-score-ppi ./designs/ \
  --checkpoint ~/.cache/boltzgen/boltz2.ckpt \
  --output ./batch_scores \
  --ligand-chains B \
  --receptor-chains A \
  --devices 4
```

### With structure refolding (comprehensive)

```bash
boltzgen-score-ppi complex.cif \
  --checkpoint ~/.cache/boltzgen/boltz2.ckpt \
  --output ./results \
  --ligand-chains B \
  --receptor-chains A \
  --with-refolding
```

This performs full structure prediction and computes RMSD and LDDT metrics by comparing the input structure to the refolded structure.

### Custom metric weights

```bash
boltzgen-score-ppi complex.cif \
  --checkpoint ~/.cache/boltzgen/boltz2.ckpt \
  --output ./scores \
  --ligand-chains B \
  --receptor-chains A \
  --metrics design_to_target_iptm=1 design_ptm=1 neg_min_design_to_target_pae=2
```

## Arguments

### Required Arguments

- `input`: Path to structure file (.cif or .pdb) or directory containing structures
- `--checkpoint`: Path to Boltz-2 checkpoint file
- `--output`: Output directory for results
- `--ligand-chains`: Comma-separated chain IDs for ligand (e.g., "B,C")
- `--receptor-chains`: Comma-separated chain IDs for receptor (e.g., "A")

### Optional Arguments

- `--metrics`: Custom metric weights (format: `metric_name=weight`)
  - Higher weight = LESS important (rank is divided by weight)
  - Default: `design_to_target_iptm=1 design_ptm=1 neg_min_design_to_target_pae=1`

- `--devices`: Number of GPU devices (default: 1)
- `--batch-size`: Batch size for prediction (default: 1)
- `--recycling-steps`: Number of recycling steps (default: 3)
- `--use-kernels`: Use GPU kernels: auto|true|false (default: auto)
- `--skip-confidence`: Skip Boltz-2 confidence prediction (faster, fewer metrics)
- `--with-refolding`: Perform structure refolding (enables RMSD/LDDT metrics)
- `--sampling-steps`: Diffusion sampling steps for refolding (default: 200)
- `--diffusion-samples`: Number of structure samples for refolding (default: 5)
- `--keep-temp`: Keep temporary directory with metadata files
- `--moldir`: Path to molecule directory (default: auto-download)
- `--cache`: Cache directory for downloads (default: ~/.cache)

## Methodology

### 1. Structure Quality Prediction

The tool runs Boltz-2 in confidence prediction mode on the input structures to compute:

- **iPTM** (interface predicted TM-score): Quality of interface contacts
- **pTM** (predicted TM-score): Overall structure quality
- **PAE** (predicted aligned error): Confidence in relative positions
- **pLDDT** (per-residue confidence scores)

### 2. Sequence Analysis

For the ligand (binding partner), the tool computes:

- Amino acid composition (fraction of each residue type)
- Sequence-based hydrophobicity
- Sequence length

### 3. Structure Refolding (Optional with `--with-refolding`)

When enabled, the tool performs full structure prediction with Boltz-2:

- **Generates multiple structure samples** (default: 5) using diffusion
- **Selects the best sample** based on confidence metrics
- **Computes RMSD metrics**:
  - `refold_rmsd`: Overall RMSD between input and refolded structure
  - `refold_rmsd_design`: RMSD for the ligand/binder portion only
  - `refold_rmsd_target`: RMSD for the receptor/target portion
  - `refold_target_aligned_rmsd_design`: Ligand RMSD after aligning on target
- **Computes LDDT metrics**:
  - `refold_lddt_intra_design`: Local structure quality of the ligand
  - `refold_lddt_inter`: Quality of interface contacts
  - `refold_lddt_all`: Overall structure quality
- **Binary designability flags**: e.g., `refold_designability_rmsd_2` (RMSD < 2Å)

These metrics validate that the input structure is physically realistic and can be accurately predicted by the model.

### 4. Rank-Based Aggregation

Following BoltzGen's filter.py methodology:

```python
# For each design:
for metric, weight in metrics.items():
    # Rank all designs by metric value (higher = better)
    rank[metric] = rank_designs_by(metric)

    # Scale rank by inverse importance weight
    scaled_rank[metric] = rank[metric] / weight

# The worst (maximum) scaled rank becomes the quality score
quality_score = max(scaled_rank.values())

# Designs are sorted by quality_score (lower is better)
final_rank = sort_by(quality_score, tiebreak_by=iptm)
```

**Key insight:** A design is only as good as its worst metric. This ensures balanced quality across all dimensions.

## Output

The tool generates a CSV file (`ppi_scores.csv`) with the following columns:

### Core Metrics
- `id`: Structure identifier
- `file_name`: Original filename
- `final_rank`: Overall rank (1 = best)
- `quality_score`: Normalized score [0, 1] (1 = best)

### Confidence Metrics (from Boltz-2)
- `design_to_target_iptm`: Interface quality
- `design_iptm`: Design chain quality
- `design_ptm`: Design structure quality
- `min_design_to_target_pae`: Minimum interface PAE
- `min_interaction_pae`: Minimum interaction PAE
- `mean_plddt`: Average per-residue confidence

### Refolding Metrics (with `--with-refolding`)
- `refold_rmsd`: Overall RMSD (Å)
- `refold_rmsd_design`: Ligand RMSD (Å)
- `refold_rmsd_target`: Receptor RMSD (Å)
- `refold_target_aligned_rmsd_design`: Ligand RMSD after target alignment (Å)
- `refold_lddt_intra_design`: Ligand internal structure quality [0-1]
- `refold_lddt_inter`: Interface quality [0-1]
- `refold_lddt_all`: Overall structure quality [0-1]
- `refold_design_to_target_iptm`: Interface iPTM from refolding
- `refold_design_ptm`: Structure pTM from refolding
- `refold_min_design_to_target_pae`: Minimum interface PAE from refolding
- Binary flags: `refold_rmsd<2.5`, `refold_designability_lddt_70`, etc.

### Sequence Metrics
- `ligand_sequence`: Amino acid sequence
- `num_ligand_residues`: Sequence length
- `ligand_hydrophobicity`: Sequence hydrophobicity
- `{AA}_fraction`: Fraction of each amino acid type

### Ranking Details
- `max_rank`: Worst scaled rank across metrics
- `secondary_rank`: Intermediate ranking value
- `rank_{metric}`: Individual metric ranks

## Examples

### Example 1: Score nanobody-antigen complexes

```bash
boltzgen-score-ppi nanobody_designs/ \
  --checkpoint ~/.cache/boltzgen/boltz2.ckpt \
  --output ./nanobody_scores \
  --ligand-chains H \
  --receptor-chains A
```

### Example 2: Multi-chain ligand

```bash
boltzgen-score-ppi complex.cif \
  --checkpoint ~/.cache/boltzgen/boltz2.ckpt \
  --output ./scores \
  --ligand-chains B,C \
  --receptor-chains A
```

### Example 3: Fast scoring (skip confidence prediction)

```bash
boltzgen-score-ppi designs/ \
  --checkpoint ~/.cache/boltzgen/boltz2.ckpt \
  --output ./scores \
  --ligand-chains B \
  --receptor-chains A \
  --skip-confidence
```

Note: With `--skip-confidence`, only sequence-based metrics are available.

### Example 4: Emphasize interface quality

```bash
boltzgen-score-ppi complex.cif \
  --checkpoint ~/.cache/boltzgen/boltz2.ckpt \
  --output ./scores \
  --ligand-chains B \
  --receptor-chains A \
  --metrics design_to_target_iptm=1 design_ptm=2 neg_min_design_to_target_pae=2
```

This gives more weight to interface quality (iptm) by making other metrics less important.

## Differences from `boltzgen-affinity`

| Feature | `boltzgen-affinity` | `boltzgen-score-ppi` |
|---------|---------------------|----------------------|
| **Scope** | Protein-small molecule | Protein-protein |
| **Method** | Neural network affinity predictor | Structure quality metrics |
| **Output** | Single affinity score | Multiple metrics + rank |
| **Refolding** | No | No |
| **Training** | Trained on binding data | Model-based predictions |

**Important:** `boltzgen-affinity` does NOT work for PPIs. It only works for protein-small molecule complexes.

## Current Limitations

### Not Yet Implemented
1. **Interface interaction metrics** (H-bonds, salt bridges): Requires feature engineering from structures
2. **Buried surface area (SASA)**: Requires special handling of atom masks

These are marked as TODO in the code and can be added in future versions.

### Performance Notes
- **Without `--with-refolding`**: Fast, relies on confidence prediction only (~10-30 seconds per structure)
- **With `--with-refolding`**: Slower, performs full structure generation (~2-5 minutes per structure depending on size and GPU)
- Refolding is recommended for final validation but not necessary for initial screening

## Future Enhancements

Planned additions (see `PPI_SCORING_METHODOLOGY.md` for details):

1. **Interface metrics computation**
   - Port `count_noncovalents()` from analyze_utils.py
   - Compute H-bonds, salt bridges directly on input structures

2. **SASA calculation**
   - Port `get_delta_sasa()` functionality
   - Requires building proper atom masks from tokenized structures

3. **Hard filters**
   - Composition filters (e.g., no Cys, not Ala-rich)
   - Sequence-based filters

4. **Diversity selection**
   - MaxMin algorithm for selecting diverse sets
   - Currently only ranking is implemented

## Troubleshooting

### "No NONPOLYMER entities found"
This error occurs if you try to use auto-detection (no --ligand-chains). For PPIs, you MUST specify both `--ligand-chains` and `--receptor-chains`.

### "Checkpoint file not found"
Download the Boltz-2 checkpoint from HuggingFace or specify the correct path with `--checkpoint`.

### GPU memory issues
- Reduce `--batch-size`
- Reduce `--recycling-steps`
- Use `--use-kernels false` to disable GPU kernels

### Slow performance
- Increase `--batch-size` if you have GPU memory
- Use `--devices` to parallelize across multiple GPUs
- Use `--skip-confidence` for very fast sequence-only scoring

## References

- **Methodology documentation:** `PPI_SCORING_METHODOLOGY.md`
- **BoltzGen filter algorithm:** `src/boltzgen/task/filter/filter.py`
- **BoltzGen analyze metrics:** `src/boltzgen/task/analyze/analyze.py`
- **Original affinity tool:** `src/boltzgen/cli/predict_affinity.py`

## Citation

If you use this tool, please cite BoltzGen:

```bibtex
@article{boltzgen2024,
  title={BoltzGen: Diffusion-based binder design},
  author={...},
  journal={...},
  year={2024}
}
```

## Contributing

To add new metrics or improve the ranking algorithm:

1. See `PPI_SCORING_METHODOLOGY.md` for implementation details
2. Metrics are computed in `compute_interface_metrics()`
3. Ranking is handled by `rank_designs()`
4. Follow the existing code structure in `analyze.py` and `filter.py`
