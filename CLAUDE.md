# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BoltzGen is a protein/peptide binder design tool built on diffusion models. It generates novel protein/peptide sequences that bind to specified target molecules. The pipeline includes:
1. **Design** - Diffusion-based structure generation
2. **Inverse Folding** - Sequence generation from backbone structures
3. **Folding/Design Folding** - Structure prediction using Boltz-2
4. **Affinity Prediction** - Binding affinity estimation (for small molecule targets)
5. **Analysis** - Quality metric computation
6. **Filtering** - Ranking and diversity selection

## Commands

### Development Setup
```bash
# Installation (requires Python >=3.9, recommended >=3.11)
pip install boltzgen

# Development installation with extra dependencies
pip install -e .[dev]
```

### Running BoltzGen

```bash
# Basic design run (downloads ~6GB models to ~/.cache on first run)
boltzgen run example/vanilla_protein/1g13prot.yaml \
  --output workbench/test_run \
  --protocol protein-anything \
  --num_designs 10 \
  --budget 2

# Check design specification before running
boltzgen check example/vanilla_peptide_with_target_binding_site/beetletert.yaml

# Run specific pipeline steps only
boltzgen run design_spec.yaml \
  --output workdir \
  --steps design inverse_folding \
  --num_designs 50

# Rerun filtering with different parameters (fast, ~15-20 seconds)
boltzgen run design_spec.yaml \
  --output workdir \
  --steps filtering \
  --refolding_rmsd_threshold 3.0 \
  --filter_biased=false

# Resume interrupted run (no progress is lost)
boltzgen run design_spec.yaml --output workdir --reuse

# Separate configure and execute for manual config editing
boltzgen configure design_spec.yaml --output workdir --protocol peptide-anything
# Edit generated YAML files in workdir/config/ if needed
boltzgen execute workdir
```

### Training Models

```bash
# Train small model (recommended for development, 8 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/boltzgen/resources/main.py \
  src/boltzgen/resources/config/train/boltzgen_small.yaml \
  name=boltzgen_small

# Train large model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/boltzgen/resources/main.py \
  src/boltzgen/resources/config/train/boltzgen.yaml \
  name=boltzgen_large

# Train inverse folding model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python src/boltzgen/resources/main.py \
  src/boltzgen/resources/config/train/inverse_folding.yaml \
  name=boltzgen_if
```

### Standalone Affinity Prediction

```bash
# Predict protein-protein affinity (specify ligand chains)
boltzgen-affinity complex.cif \
  --checkpoint ~/.cache/huggingface/.../boltz2_aff.ckpt \
  --output ./results \
  --ligand-chains B,C

# Specify both ligand and receptor explicitly
boltzgen-affinity complex.cif \
  --checkpoint affinity.ckpt \
  --output ./results \
  --ligand-chains B,C \
  --receptor-chains A

# Specify receptor, infer ligand from remaining chains
boltzgen-affinity designs/ \
  --checkpoint affinity.ckpt \
  --output ./results \
  --receptor-chains A

# Protein-small molecule (auto-detects NONPOLYMER as ligand)
boltzgen-affinity protein_ligand.cif \
  --checkpoint affinity.ckpt \
  --output ./results

# With custom settings
boltzgen-affinity designs/ \
  --checkpoint affinity.ckpt \
  --output ./affinity_results \
  --ligand-chains B \
  --devices 2 \
  --batch-size 4 \
  --keep-temp
```

**Notes**:
- For protein-protein affinity, specify `--ligand-chains` (binding partner) and optionally `--receptor-chains`
- If only ligand specified, all other protein chains are treated as receptor
- If only receptor specified, all other protein chains are treated as ligand
- For protein-small molecule, NONPOLYMER entities are auto-detected as ligand
- Metadata files are generated automatically from structures (no pre-processing needed)
- Use `--keep-temp` to preserve generated metadata files for inspection

### Testing
No test suite is present in the repository. Testing is done through example runs.

### Linting
```bash
# The project uses ruff for linting (configured in pyproject.toml)
ruff check src/
ruff format src/
```

## Architecture

### Code Organization

```
src/boltzgen/
├── cli/              # CLI entry point (boltzgen.py)
├── data/             # Data processing, parsing, featurization
│   ├── parse/        # mmCIF/PDB/A3M parsers, YAML design spec parser
│   ├── feature/      # Feature computation (MSA, templates, etc.)
│   ├── crop/         # Training data cropping
│   ├── filter/       # Data filtering logic
│   └── write/        # mmCIF/PDB writers
├── model/            # Neural network models
│   ├── models/       # Main Boltz model (boltz.py)
│   ├── modules/      # Model components (diffusion, inverse_fold, affinity, confidence)
│   ├── layers/       # Neural network layers (pairformer, attention, etc.)
│   ├── loss/         # Loss functions
│   └── optim/        # Optimizers and schedulers
├── task/             # Pipeline tasks (executable units)
│   ├── predict/      # GPU inference tasks (design, folding, affinity)
│   ├── analyze/      # CPU analysis task
│   ├── filter/       # CPU filtering/ranking task
│   └── train/        # Training task
└── resources/        # Configuration files and main.py entry point
    ├── main.py       # Task runner (called by CLI for each pipeline step)
    └── config/       # YAML configs for each pipeline step
```

### Pipeline Execution Flow

1. **CLI (`src/boltzgen/cli/boltzgen.py`)**:
   - User runs `boltzgen run design_spec.yaml --output workdir`
   - `configure_command()` creates `BinderDesignPipeline`, writes resolved configs to `workdir/config/<step>.yaml` and `workdir/steps.yaml`
   - `execute_command()` launches each step as subprocess: `python src/boltzgen/resources/main.py workdir/config/<step>.yaml`

2. **Task Runner (`src/boltzgen/resources/main.py`)**:
   - Loads YAML config, instantiates the Task class (via Hydra)
   - Calls `task.run(config)` for execution

3. **Task Classes**:
   - **Predict** (`task/predict/predict.py`): GPU tasks (design, inverse_folding, folding, design_folding, affinity)
   - **Analyze** (`task/analyze/analyze.py`): CPU metrics computation
   - **Filter** (`task/filter/filter.py`): Ranking and diversity selection

### Key Model Components

- **Boltz** (`model/models/boltz.py`): Main LightningModule containing all model components
- **AtomDiffusion** (`model/modules/diffusion.py`): Diffusion model for structure generation
- **InverseFoldingEncoder/Decoder** (`model/modules/inverse_fold.py`): Sequence design from structure
- **ConfidenceModule** (`model/modules/confidence.py`): Structure quality prediction
- **AffinityModule** (`model/modules/affinity.py`): Binding affinity prediction
- **Pairformer** (`model/layers/pairformer.py`): Core attention-based architecture

### Data Processing

- **YamlDesignParser** (`data/parse/schema.py`): Parses design specification YAMLs into Structure objects
- **Structure**: Central data structure (defined in `data/parse/mmcif.py`) containing:
  - Chains, residues, atoms with coordinates and metadata
  - Design masks (what to design), binding types, structure groups
- **Featurizer** (`data/feature/featurizer.py`): Converts Structure to model input features
- **Writers** (`task/predict/writer.py`): Save generated designs to mmCIF/NPZ files

### Protocols

Protocols determine which steps run and default parameters:
- **protein-anything**: Design proteins to bind proteins/peptides (includes design_folding)
- **peptide-anything**: Design (cyclic) peptides (no Cys in inverse folding, no design_folding)
- **protein-small_molecule**: Design proteins to bind small molecules (includes affinity prediction)
- **nanobody-anything**: Design nanobodies (no Cys, no design_folding)

Protocol-specific config overrides are in `cli/boltzgen.py:protocol_configs`.

## Important Details

### Design Specification YAMLs

- All residue indices are **1-indexed** and use mmCIF `label_asym_id` (NOT `auth_asym_id`)
- File paths in YAMLs are relative to the YAML file location
- Key concepts:
  - **entities**: Define targets (from .cif files) and designed molecules (proteins/peptides/ligands)
  - **binding_types**: Specify where designs should/shouldn't bind (`binding`, `not_binding`)
  - **structure_groups**: Control structural constraints (visibility levels)
  - **design**: Mark residues to redesign in existing structures
  - **constraints**: Define bonds (disulfide bridges, staples)

### Checkpoints and Artifacts

Models are hosted on HuggingFace and downloaded automatically:
- Format: `huggingface:<repo_id>:<filename>`
- Default cache: `~/.cache` (override with `--cache` or `$HF_HOME`)
- Artifacts defined in `cli/boltzgen.py:ARTIFACTS`

### GPU Optimization

- Kernels (cuequivariance) used on devices with compute capability >= 8.0
- Controlled via `--use_kernels {auto,true,false}`
- Environment variables set in predict.py: `CUEQ_DEFAULT_CONFIG`, `CUEQ_DISABLE_AOT_TUNING`

### Output Directory Structure

```
output_dir/
├── config/                              # Pipeline step configurations
├── steps.yaml                           # Pipeline manifest
├── intermediate_designs/                # Design step output (CIF/NPZ)
├── intermediate_designs_inverse_folded/ # After inverse folding
│   ├── refold_cif/                      # Refolded complexes (target + binder)
│   ├── refold_design_cif/               # Refolded binders only
│   ├── aggregate_metrics_analyze.csv    # Aggregated metrics
│   └── per_target_metrics_analyze.csv   # Per-design metrics
└── final_ranked_designs/                # Filtering output
    ├── intermediate_ranked_<N>_designs/ # Top-N by quality
    ├── final_<budget>_designs/          # Quality + diversity set
    ├── all_designs_metrics.csv
    ├── final_designs_metrics_<budget>.csv
    └── results_overview.pdf
```

### Filtering Metrics

- Filtering uses rank-based aggregation of multiple metrics (lower rank = better)
- Default metrics vary by protocol (see `task/filter/filter.py`)
- Override with `--metrics_override metric_name=weight` (higher weight = less important)
- Hard filters with `--additional_filters 'feature<threshold'` (quote to avoid shell issues)
- Diversity selection via MaxMin algorithm with `--alpha` (0.0=quality only, 1.0=diversity only)

### Training Data Requirements

Training requires (see README "Training BoltzGen models"):
- `targets/` - Training structure dataset
- `msa/` - Multiple sequence alignments
- `mols/` - Small molecule dictionary
- `boltz2_fold.ckpt` - Folding model checkpoint
- Optionally: `boltzgen1_structuretrained_small.ckpt` for pretraining

Update paths in `src/boltzgen/resources/config/train/*.yaml` if not using default `./training_data/`.

## Common Pitfalls

1. **Residue indexing**: Always use 1-indexed `label_asym_id`, verify in https://molstar.org/viewer/
2. **File paths in YAMLs**: Must be relative to YAML file location, not current working directory
3. **Interrupted runs**: Use `--reuse` to resume without losing progress
4. **Filtering tuning**: Run filtering step repeatedly with `--steps filtering` (very fast) to tune thresholds
5. **Batch size for length diversity**: Large `--diffusion_batch_size` relative to `--num_designs` reduces length sampling diversity (designs in same batch share random parameters)
6. **Inverse folding restrictions**: Set `--inverse_fold_avoid ''` (empty string) if you want Cysteines in peptide/nanobody protocols
7. **Subprocess mode**: Don't use `--no_subprocess` with `--devices > 1` (causes issues)
