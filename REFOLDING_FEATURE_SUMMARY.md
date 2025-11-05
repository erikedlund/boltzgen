# Refolding Feature Added to PPI Scoring Tool

## Overview

The `boltzgen-score-ppi` tool now supports optional structure refolding with the `--with-refolding` flag. This enables computation of RMSD and LDDT metrics by comparing input structures to Boltz-2 predictions.

## What Was Added

### New CLI Arguments

1. **`--with-refolding`**: Enable structure refolding (default: disabled)
2. **`--sampling-steps`**: Number of diffusion sampling steps (default: 200)
3. **`--diffusion-samples`**: Number of structure samples to generate (default: 5)

### New Metrics Computed

When `--with-refolding` is enabled, the tool computes:

#### RMSD Metrics
- `refold_rmsd`: Overall RMSD between input and refolded structure
- `refold_rmsd_design`: RMSD for ligand/binder only
- `refold_rmsd_target`: RMSD for receptor/target only
- `refold_rmsd_design_target`: Combined design+target RMSD
- `refold_target_aligned_rmsd_design`: Ligand RMSD after aligning on target
- `neg_refold_rmsd`: Negative RMSD (for ranking, higher is better)
- `neg_refold_rmsd_design`: Negative ligand RMSD

#### LDDT Metrics
- `refold_lddt_intra_design`: Local structure quality of ligand
- `refold_lddt_inter`: Quality of interface contacts
- `refold_lddt_all`: Overall structure quality
- `refold_lddt_intra_target`: Local structure quality of receptor

#### Confidence Metrics (from refolding)
- `refold_design_to_target_iptm`: Interface iPTM from refolded structure
- `refold_design_iptm`: Design chain iPTM
- `refold_design_ptm`: Design structure pTM
- `refold_target_ptm`: Target structure pTM
- `refold_ptm`: Overall pTM
- `refold_min_design_to_target_pae`: Minimum interface PAE
- `refold_min_interaction_pae`: Minimum interaction PAE

#### Binary Designability Flags
- `refold_rmsd<2.5`: RMSD below 2.5Å threshold
- `refold_target_aligned<2.5`: Target-aligned RMSD below 2.5Å
- `refold_designability_rmsd_2`: Ligand RMSD below 2.0Å
- `refold_designability_rmsd_4`: Ligand RMSD below 4.0Å
- `refold_designability_lddt_60` through `refold_designability_lddt_90`: LDDT thresholds
- `refold_design_ptm>80`, `refold_design_ptm>75`: pTM thresholds
- `refold_design_iptm>80`, `refold_design_iptm>70`, etc.: iPTM thresholds
- `refold_min_interaction_pae<1.5` through `refold_min_interaction_pae<5`: PAE thresholds

### Updated Default Metrics for Ranking

When `--with-refolding` is enabled, the default metrics configuration includes:

```python
{
    "design_to_target_iptm": 1,        # Interface quality
    "design_ptm": 1,                    # Structure quality
    "neg_min_design_to_target_pae": 1,  # Interface confidence
    "neg_refold_rmsd": 1,               # Refolding accuracy (NEW)
    "neg_refold_rmsd_design": 1,        # Ligand refolding accuracy (NEW)
    "refold_lddt_intra_design": 2,      # Ligand local quality (NEW)
}
```

Users can override with custom weights using `--metrics`.

## Implementation Details

### Code Changes

#### 1. New Function: `create_refolding_config()`
**Location:** `src/boltzgen/cli/score_ppi.py:387-493`

Creates a Boltz-2 configuration for full structure refolding:
- Uses `FoldingWriter` to save results to `fold_out_npz/` directory
- Enables template usage for better folding
- Configurable sampling steps and diffusion samples
- Returns all confidence metrics

#### 2. Updated Function: `compute_interface_metrics()`
**Location:** `src/boltzgen/cli/score_ppi.py:496-636`

Added refolding metrics computation:
- Loads refolded structure from `fold_out_npz/{structure_id}.npz`
- Builds feature dictionary from tokenized structure
- Calls `get_fold_metrics()` to compute RMSD, LDDT, and designability flags
- Prefixes all metrics with `refold_` to distinguish from confidence-only metrics
- Handles errors gracefully with warnings

#### 3. Updated Main Execution Flow
**Location:** `src/boltzgen/cli/score_ppi.py:956-990`

Added refolding step between confidence prediction and metrics computation:
1. Confidence prediction (if not skipped)
2. **Refolding (NEW, if `--with-refolding` enabled)**
3. Interface metrics computation (now includes refolding metrics)
4. Ranking and output

### Dependencies

The refolding feature reuses existing BoltzGen components:
- `boltzgen.task.predict.predict.Predict`: Task runner
- `boltzgen.task.predict.writer.FoldingWriter`: Saves refolding results
- `boltzgen.task.analyze.analyze_utils.get_fold_metrics()`: Computes RMSD/LDDT
- `boltzgen.task.analyze.analyze_utils.get_best_folding_sample()`: Selects best sample
- `boltzgen.data.const.folding_dirname`: Standard output directory name

No new dependencies were added.

## Usage Examples

### Basic Refolding

```bash
boltzgen-score-ppi complex.cif \
  --checkpoint ~/.cache/boltzgen/boltz2.ckpt \
  --output ./results \
  --ligand-chains B \
  --receptor-chains A \
  --with-refolding
```

### Refolding with Custom Sampling

```bash
boltzgen-score-ppi designs/ \
  --checkpoint ~/.cache/boltzgen/boltz2.ckpt \
  --output ./results \
  --ligand-chains B \
  --receptor-chains A \
  --with-refolding \
  --sampling-steps 300 \
  --diffusion-samples 10
```

### Custom Metric Weights with Refolding

```bash
boltzgen-score-ppi complex.cif \
  --checkpoint ~/.cache/boltzgen/boltz2.ckpt \
  --output ./results \
  --ligand-chains B \
  --receptor-chains A \
  --with-refolding \
  --metrics design_to_target_iptm=1 neg_refold_rmsd=1 refold_lddt_intra_design=3
```

This emphasizes interface quality and refolding accuracy over ligand local quality.

## Output

With `--with-refolding`, the output CSV includes all refolding metrics:

```csv
id,final_rank,quality_score,design_to_target_iptm,design_ptm,refold_rmsd,refold_rmsd_design,refold_lddt_intra_design,...
complex1,1,1.0,0.85,0.90,1.2,0.8,0.92,...
complex2,2,0.75,0.82,0.88,2.1,1.5,0.85,...
...
```

## Performance Considerations

### Computational Cost

- **Without refolding**: ~10-30 seconds per structure
- **With refolding**: ~2-5 minutes per structure (depends on size, GPU, sampling parameters)

### Recommendations

1. **Initial screening**: Use without `--with-refolding` for fast scoring
2. **Final validation**: Use `--with-refolding` on top candidates
3. **GPU memory**: Reduce `--diffusion-samples` if running out of memory
4. **Batch processing**: Use `--devices > 1` for parallel processing

### Sampling Parameters

- **Fewer samples** (e.g., `--diffusion-samples 1`): Faster, less robust
- **More samples** (e.g., `--diffusion-samples 10`): Slower, more robust
- **Fewer steps** (e.g., `--sampling-steps 100`): Faster, lower quality
- **More steps** (e.g., `--sampling-steps 500`): Slower, higher quality

Default values (200 steps, 5 samples) balance speed and quality.

## Validation

The refolding metrics validate that input structures are:

1. **Physically realistic**: Low RMSD indicates the structure can be accurately predicted
2. **Well-folded**: High LDDT indicates good local geometry
3. **Stable interfaces**: Low interface PAE indicates confident binding geometry

Structures with:
- `refold_rmsd < 2.5Å`: Generally well-predicted
- `refold_lddt_intra_design > 0.7`: Good local structure quality
- `refold_design_to_target_iptm > 0.6`: Confident interface

are considered high quality.

## Future Enhancements

Potential additions:
1. **Isolated ligand refolding**: Refold ligand without receptor (like `design_folding` step)
2. **SASA metrics from refolded structures**: Compute buried surface area
3. **Interface interaction metrics from refolded structures**: H-bonds, salt bridges
4. **Diversity metrics**: Compare structural diversity across refolded samples

## Testing

To test the refolding feature:

```bash
# 1. Prepare a test PPI structure
boltzgen-score-ppi test_complex.cif \
  --checkpoint ~/.cache/boltzgen/boltz2.ckpt \
  --output ./test_results \
  --ligand-chains B \
  --receptor-chains A \
  --with-refolding \
  --keep-temp

# 2. Check outputs
ls ./test_results/
# Expected: ppi_scores.csv

ls <temp_dir>/fold_out_npz/
# Expected: test_complex.npz

ls <temp_dir>/refold_cif/
# Expected: test_complex.cif (refolded structure)

# 3. Verify metrics in CSV
head -n 2 ./test_results/ppi_scores.csv | cut -d',' -f1-10
# Should include refold_rmsd, refold_lddt_intra_design, etc.
```

## Documentation Updates

Updated files:
1. **`src/boltzgen/cli/score_ppi.py`**: Added refolding implementation
2. **`PPI_SCORING_TOOL_README.md`**: Added refolding documentation
3. **`REFOLDING_FEATURE_SUMMARY.md`**: This file

## Backward Compatibility

The refolding feature is **fully backward compatible**:
- Default behavior unchanged (no refolding)
- Existing scripts work without modification
- Refolding is opt-in via `--with-refolding` flag

## Summary

The refolding feature provides comprehensive structure validation for PPI scoring. It's recommended for final validation but optional for initial screening. The implementation reuses existing BoltzGen infrastructure and integrates seamlessly with the rank-based aggregation algorithm.
