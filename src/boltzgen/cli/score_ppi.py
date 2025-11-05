#!/usr/bin/env python3
"""
Standalone PPI scoring script for BoltzGen.

This script scores protein-protein interactions using the same methodology
as BoltzGen's nanobody/protein-anything workflows, but without requiring
the full design pipeline. It computes:

1. Structure quality metrics (iPTM, PAE, pLDDT) via Boltz-2 confidence prediction
2. Interface analysis (H-bonds, salt bridges, buried surface area)
3. Rank-based aggregation scoring across multiple metrics

Unlike boltzgen-affinity (which only works for protein-small molecule),
this tool properly handles protein-protein complexes.

Usage:
    boltzgen-score-ppi INPUT_PATH --checkpoint CKPT_PATH --output OUTPUT_DIR [OPTIONS]

Examples:
    # Score a single PPI complex
    boltzgen-score-ppi complex.cif --checkpoint boltz2.ckpt --output ./results \\
        --ligand-chains B --receptor-chains A

    # Score multiple complexes in a directory
    boltzgen-score-ppi designs/ --checkpoint boltz2.ckpt --output ./batch_scores \\
        --ligand-chains B --devices 4

    # With structure refolding (for RMSD metrics)
    boltzgen-score-ppi complex.cif --checkpoint boltz2.ckpt --output ./scores \\
        --ligand-chains B --receptor-chains A \\
        --with-refolding

    # Custom metric weights
    boltzgen-score-ppi complex.cif --checkpoint boltz2.ckpt --output ./scores \\
        --ligand-chains B \\
        --metrics design_to_target_iptm=1 neg_refold_rmsd=1 refold_lddt_intra_design=2
"""

from boltzgen.utils.quiet import quiet_startup

quiet_startup()

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Set, Dict
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from boltzgen.task.predict.predict import Predict
from boltzgen.task.predict.data_from_generated import FromGeneratedDataModule
from boltzgen.data import const
from boltzgen.data.parse import mmcif
from boltzgen.data.parse.pdb_parser import parse_pdb
from boltzgen.data.mol import load_canonicals
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.task.analyze.analyze_utils import (
    count_noncovalents,
    get_delta_sasa,
    calc_hydrophobicity,
    get_fold_metrics,
    get_best_folding_sample,
)
from boltzgen.task.predict.data_from_generated import collate


def generate_metadata_for_ppi(
    structure_path: Path,
    moldir: Path,
    ligand_chains: Set[str],
    receptor_chains: Set[str],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate .npz metadata file for a PPI structure.

    Parameters
    ----------
    structure_path : Path
        Path to structure file (.cif or .pdb)
    moldir : Path
        Path to molecule directory
    ligand_chains : Set[str]
        Set of chain IDs to treat as ligand (binding partner)
    receptor_chains : Set[str]
        Set of chain IDs to treat as receptor
    output_path : Optional[Path]
        Where to save the metadata file. If None, saves next to structure file.

    Returns
    -------
    Path
        Path to generated metadata file
    """
    # Parse structure
    canonicals = load_canonicals(moldir)

    try:
        if structure_path.suffix == ".cif":
            parsed = mmcif.parse_mmcif(
                structure_path, canonicals, moldir=moldir, use_original_res_idx=False
            )
        elif structure_path.suffix == ".pdb":
            parsed = parse_pdb(
                structure_path, moldir=moldir, use_original_res_idx=False
            )
        else:
            raise ValueError(f"Unsupported file format: {structure_path.suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to parse {structure_path}: {e}") from e

    structure = parsed.data

    # Tokenize structure
    tokenizer = Tokenizer(atomize_modified_residues=False)
    try:
        tokenized = tokenizer.tokenize(structure, inverse_fold=False)
    except Exception as e:
        raise RuntimeError(f"Failed to tokenize {structure_path}: {e}") from e

    num_tokens = len(tokenized.tokens)

    # Get available chains
    available_chains = set(chain["name"] for chain in structure.chains)
    chain_ids = tokenized.tokens["asym_id"].numpy()
    chain_names = [structure.chains[cid]["name"] for cid in chain_ids]

    # Validate specified chains exist
    missing_ligand = ligand_chains - available_chains
    if missing_ligand:
        raise ValueError(
            f"Specified ligand chains {missing_ligand} not found in structure. "
            f"Available chains: {available_chains}"
        )

    missing_receptor = receptor_chains - available_chains
    if missing_receptor:
        raise ValueError(
            f"Specified receptor chains {missing_receptor} not found in structure. "
            f"Available chains: {available_chains}"
        )

    # Check for overlap
    overlap = ligand_chains & receptor_chains
    if overlap:
        raise ValueError(
            f"Chains {overlap} specified as both ligand and receptor. "
            "Each chain must be either ligand or receptor, not both."
        )

    # Create masks
    ligand_token_mask = np.array(
        [name in ligand_chains for name in chain_names], dtype=bool
    )
    receptor_token_mask = np.array(
        [name in receptor_chains for name in chain_names], dtype=bool
    )

    if not ligand_token_mask.any():
        raise ValueError(
            f"No tokens found for ligand chains {ligand_chains}. "
            f"Available chains: {available_chains}"
        )
    if not receptor_token_mask.any():
        raise ValueError(
            f"No tokens found for receptor chains {receptor_chains}. "
            f"Available chains: {available_chains}"
        )

    print(f"  Ligand chains: {ligand_chains}")
    print(f"  Receptor chains: {receptor_chains}")
    print(f"  Total tokens: {num_tokens} (Receptor: {receptor_token_mask.sum()}, Ligand: {ligand_token_mask.sum()})")

    # Initialize metadata arrays
    design_mask = ligand_token_mask.astype(np.float32)  # Mark ligand as "design" for interface metrics
    mol_type = tokenized.tokens["mol_type"].numpy()
    ss_type = np.zeros(num_tokens, dtype=np.int64)
    binding_type = np.zeros(num_tokens, dtype=np.int64)
    token_resolved_mask = np.ones(num_tokens, dtype=bool)

    # Save metadata
    if output_path is None:
        output_path = structure_path.with_suffix(".npz")

    np.savez_compressed(
        output_path,
        design_mask=design_mask,
        mol_type=mol_type,
        ss_type=ss_type,
        binding_type=binding_type,
        token_resolved_mask=token_resolved_mask,
        ligand_token_mask=ligand_token_mask,
        receptor_token_mask=receptor_token_mask,
    )

    return output_path


def prepare_structures_for_ppi_scoring(
    input_path: Path,
    moldir: Path,
    ligand_chains: List[str],
    receptor_chains: List[str],
    temp_dir: Optional[Path] = None,
) -> Path:
    """
    Prepare structure(s) for PPI scoring by generating metadata.

    Parameters
    ----------
    input_path : Path
        Input structure file or directory
    moldir : Path
        Path to molecule directory
    ligand_chains : List[str]
        List of chain IDs to treat as ligand
    receptor_chains : List[str]
        List of chain IDs to treat as receptor
    temp_dir : Optional[Path]
        Temporary directory to use. If None, creates one.

    Returns
    -------
    Path
        Directory containing structures with metadata
    """
    ligand_chain_set = set(ligand_chains)
    receptor_chain_set = set(receptor_chains)

    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="boltzgen_ppi_scoring_"))

    print(f"\nPreparing structures in: {temp_dir}")

    # Collect structure files
    if input_path.is_file():
        structure_files = [input_path]
    else:
        structure_files = list(input_path.glob("*.cif")) + list(input_path.glob("*.pdb"))
        structure_files = [f for f in structure_files if "_native" not in f.name]

    if not structure_files:
        raise FileNotFoundError(f"No structure files found in {input_path}")

    print(f"Found {len(structure_files)} structure(s) to process\n")

    # Process each structure
    failed = []
    for struct_file in structure_files:
        print(f"Processing: {struct_file.name}")

        # Copy structure to temp directory
        dest_struct = temp_dir / struct_file.name
        shutil.copy2(struct_file, dest_struct)

        # Generate metadata
        try:
            metadata_path = generate_metadata_for_ppi(
                dest_struct, moldir, ligand_chain_set, receptor_chain_set
            )
            print(f"  Generated metadata: {metadata_path.name}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")
            failed.append((struct_file.name, str(e)))
            # Remove the copied structure file if metadata generation failed
            dest_struct.unlink()

    if failed:
        print("\nFailed to process the following structures:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        print()

    # Check if any structures were successfully processed
    processed_structures = list(temp_dir.glob("*.cif")) + list(temp_dir.glob("*.pdb"))
    if not processed_structures:
        raise RuntimeError("Failed to process any structures. See errors above.")

    print(f"Successfully prepared {len(processed_structures)}/{len(structure_files)} structure(s)")

    return temp_dir


def create_confidence_prediction_config(
    prepared_dir: Path,
    checkpoint: Path,
    output: Path,
    devices: int = 1,
    batch_size: int = 1,
    num_workers: int = 4,
    moldir: str = None,
    recycling_steps: int = 3,
    use_kernels: str = "auto",
) -> OmegaConf:
    """
    Create configuration for Boltz-2 confidence prediction (no structure generation).

    Parameters
    ----------
    prepared_dir : Path
        Directory containing prepared structures with metadata
    checkpoint : Path
        Path to the Boltz-2 checkpoint
    output : Path
        Output directory for predictions
    devices : int
        Number of GPU devices to use
    batch_size : int
        Batch size for prediction
    num_workers : int
        Number of dataloader workers
    moldir : str
        Path to molecule directory
    recycling_steps : int
        Number of recycling steps for the model
    use_kernels : str
        Whether to use kernels ('auto', 'true', 'false')

    Returns
    -------
    OmegaConf
        Configuration object for confidence prediction
    """
    # Create configuration for folding/confidence prediction
    config = OmegaConf.create(
        {
            "_target_": "boltzgen.task.predict.predict.Predict",
            "debug": False,
            "data": {
                "_target_": "boltzgen.task.predict.data_from_generated.FromGeneratedDataModule",
                "cfg": {
                    "_target_": "boltzgen.task.predict.data_from_generated.DataConfig",
                    "tokenizer": {
                        "_target_": "boltzgen.data.tokenize.tokenizer.Tokenizer",
                        "atomize_modified_residues": False,
                    },
                    "featurizer": {
                        "_target_": "boltzgen.data.feature.featurizer.Featurizer",
                    },
                    "suffix": ".cif",
                    "suffix_metadata": ".npz",
                    "suffix_native": "_native.cif",
                    "samples_per_target": 1000000000000000,
                    "num_targets": 10000000000000,
                    "moldir": moldir,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "pin_memory": True,
                },
                "design_dir": str(prepared_dir),
                "return_native": False,
                "compute_affinity": False,  # NOT using affinity module
                "target_templates": False,
                "fail_if_no_designs": True,
            },
            "keys_dict_out": const.eval_keys_confidence,  # iPTM, PAE, pLDDT, etc.
            "writer": {
                "_target_": "boltzgen.task.predict.writer.ConfidenceWriter",
                "design_dir": str(prepared_dir),
            },
            "trainer": {
                "accelerator": "gpu",
                "logger": False,
                "devices": devices,
                "precision": "bf16-mixed",
            },
            "name": "ppi_confidence_prediction",
            "output": str(output),
            "checkpoint": str(checkpoint),
            "matmul_precision": None,
            "recycling_steps": recycling_steps,
            "sampling_steps": 0,  # No diffusion sampling
            "diffusion_samples": 1,  # Just one forward pass
            "override": {
                "validators": None,
                "use_kernels": use_kernels == "true"
                if use_kernels in ["true", "false"]
                else None,
            },
        }
    )

    return config


def create_refolding_config(
    prepared_dir: Path,
    checkpoint: Path,
    output: Path,
    devices: int = 1,
    batch_size: int = 1,
    num_workers: int = 4,
    moldir: str = None,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 5,
    use_kernels: str = "auto",
) -> OmegaConf:
    """
    Create configuration for Boltz-2 refolding (full structure prediction).

    Parameters
    ----------
    prepared_dir : Path
        Directory containing prepared structures with metadata
    checkpoint : Path
        Path to the Boltz-2 checkpoint
    output : Path
        Output directory for predictions
    devices : int
        Number of GPU devices to use
    batch_size : int
        Batch size for prediction
    num_workers : int
        Number of dataloader workers
    moldir : str
        Path to molecule directory
    recycling_steps : int
        Number of recycling steps for the model
    sampling_steps : int
        Number of diffusion sampling steps
    diffusion_samples : int
        Number of structure samples to generate
    use_kernels : str
        Whether to use kernels ('auto', 'true', 'false')

    Returns
    -------
    OmegaConf
        Configuration object for refolding
    """
    # Create configuration for refolding (similar to fold.yaml)
    config = OmegaConf.create(
        {
            "_target_": "boltzgen.task.predict.predict.Predict",
            "debug": False,
            "data": {
                "_target_": "boltzgen.task.predict.data_from_generated.FromGeneratedDataModule",
                "cfg": {
                    "_target_": "boltzgen.task.predict.data_from_generated.DataConfig",
                    "tokenizer": {
                        "_target_": "boltzgen.data.tokenize.tokenizer.Tokenizer",
                        "atomize_modified_residues": False,
                    },
                    "featurizer": {
                        "_target_": "boltzgen.data.feature.featurizer.Featurizer",
                    },
                    "suffix": ".cif",
                    "suffix_metadata": ".npz",
                    "suffix_native": "_native.cif",
                    "samples_per_target": 1000000000000000,
                    "num_targets": 10000000000000,
                    "moldir": moldir,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "pin_memory": True,
                },
                "design_dir": str(prepared_dir),
                "return_native": False,
                "compute_affinity": False,
                "target_templates": True,  # Use templates for better folding
                "fail_if_no_designs": True,
                "output_dir": None,  # Will use design_dir/fold_out_npz
            },
            "keys_dict_out": const.eval_keys_confidence,  # Include all confidence metrics
            "writer": {
                "_target_": "boltzgen.task.predict.writer.FoldingWriter",
                "design_dir": str(prepared_dir),
            },
            "trainer": {
                "accelerator": "gpu",
                "logger": False,
                "devices": devices,
                "precision": "bf16-mixed",
            },
            "name": "ppi_refolding",
            "output": str(output),
            "checkpoint": str(checkpoint),
            "matmul_precision": None,
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "override": {
                "validators": None,
                "use_kernels": use_kernels == "true"
                if use_kernels in ["true", "false"]
                else None,
            },
        }
    )

    return config


def compute_interface_metrics(
    structure_path: Path,
    metadata_path: Path,
    confidence_path: Optional[Path] = None,
    refolding_path: Optional[Path] = None,
    moldir: Optional[Path] = None,
) -> Dict:
    """
    Compute interface metrics for a PPI structure.

    Parameters
    ----------
    structure_path : Path
        Path to structure file
    metadata_path : Path
        Path to metadata .npz file
    confidence_path : Optional[Path]
        Path to confidence predictions .npz file
    refolding_path : Optional[Path]
        Path to refolding predictions .npz file
    moldir : Optional[Path]
        Path to molecule directory

    Returns
    -------
    Dict
        Dictionary of computed metrics
    """
    metrics = {}
    metrics["id"] = structure_path.stem
    metrics["file_name"] = structure_path.name

    # Load metadata
    metadata = np.load(metadata_path)
    design_mask = metadata["design_mask"]
    ligand_mask = metadata["ligand_token_mask"]

    # Parse structure
    canonicals = load_canonicals(None)  # Will use default
    if structure_path.suffix == ".cif":
        parsed = mmcif.parse_mmcif(structure_path, canonicals, use_original_res_idx=False)
    else:
        parsed = parse_pdb(structure_path, use_original_res_idx=False)

    structure = parsed.data
    tokenizer = Tokenizer(atomize_modified_residues=False)
    tokenized = tokenizer.tokenize(structure, inverse_fold=False)

    # Get ligand sequence
    res_type_argmax = torch.argmax(tokenized.tokens["res_type"], dim=-1)
    ligand_seq_tensor = res_type_argmax[torch.from_numpy(ligand_mask)]
    ligand_seq = "".join(
        [const.prot_token_to_letter.get(const.tokens[t], "X") for t in ligand_seq_tensor]
    )
    metrics["ligand_sequence"] = ligand_seq
    metrics["num_ligand_residues"] = len(ligand_seq)

    # Amino acid composition
    for aa in const.fake_atom_placements.keys():
        metrics[f"{aa}_fraction"] = (
            (ligand_seq_tensor == const.token_ids[aa]).float().mean().item()
        )

    # Hydrophobicity
    metrics["ligand_hydrophobicity"] = calc_hydrophobicity(ligand_seq)

    # Load confidence predictions if available
    if confidence_path and confidence_path.exists():
        confidence = np.load(confidence_path)
        for key in const.eval_keys_confidence:
            if key in confidence:
                metrics[key] = float(confidence[key])

        # Create derived metrics
        if "min_interaction_pae" in metrics:
            metrics["neg_min_interaction_pae"] = -metrics["min_interaction_pae"]
        if "min_design_to_target_pae" in metrics:
            metrics["neg_min_design_to_target_pae"] = -metrics["min_design_to_target_pae"]

    # Compute refolding metrics if available
    if refolding_path and refolding_path.exists():
        try:
            # Load refolding results
            folded = np.load(refolding_path)

            # Create features dict from tokenized structure
            # We need to convert to the format expected by get_fold_metrics
            feat = {}
            for key in ["res_type", "mol_type", "asym_id", "entity_id"]:
                if key in tokenized.tokens:
                    feat[key] = tokenized.tokens[key]

            # Add coordinates from original structure
            feat["coords"] = tokenized.coords.unsqueeze(0)  # Add batch dimension

            # Add design mask (ligand is the "design")
            feat["design_mask"] = torch.from_numpy(design_mask).bool()

            # Add atom-level features
            if hasattr(tokenized, "atom_pad_mask"):
                feat["atom_pad_mask"] = tokenized.atom_pad_mask
            if hasattr(tokenized, "atom_resolved_mask"):
                feat["atom_resolved_mask"] = tokenized.atom_resolved_mask
            if hasattr(tokenized, "atom_to_token"):
                feat["atom_to_token"] = tokenized.atom_to_token
            if hasattr(tokenized, "token_to_rep_atom"):
                feat["token_to_rep_atom"] = tokenized.token_to_rep_atom
            if hasattr(tokenized, "backbone_mask"):
                feat["backbone_mask"] = tokenized.backbone_mask

            # Compute refolding metrics (RMSD, LDDT, etc.)
            fold_metrics = get_fold_metrics(
                feat,
                folded,
                compute_lddts=True,
                prefix="",  # No prefix for all-atom metrics
            )

            # Add fold metrics to results
            for key, value in fold_metrics.items():
                metrics[f"refold_{key}"] = value

            # Create derived metrics for ranking
            if "refold_rmsd" in metrics:
                metrics["neg_refold_rmsd"] = -metrics["refold_rmsd"]
            if "refold_rmsd_design" in metrics:
                metrics["neg_refold_rmsd_design"] = -metrics["refold_rmsd_design"]

        except Exception as e:
            print(f"Warning: Failed to compute refolding metrics for {structure_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # TODO: Compute interface interactions (H-bonds, salt bridges)
    # This requires creating features from the structure
    # For now, we'll skip this and add it in a follow-up

    # TODO: Compute SASA
    # This also requires special handling

    return metrics


def rank_designs(df: pd.DataFrame, metrics_config: Dict[str, float]) -> pd.DataFrame:
    """
    Rank designs using the same rank-based aggregation as BoltzGen filter.py.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with computed metrics
    metrics_config : Dict[str, float]
        Dictionary mapping metric names to inverse importance weights
        Higher weight = less important (rank is divided by weight)

    Returns
    -------
    pd.DataFrame
        DataFrame with ranking columns added
    """
    # Create rank dataframe
    rank_df = pd.DataFrame(index=df.index)

    # For each metric, compute scaled rank
    for col, inverse_importance in metrics_config.items():
        if col not in df.columns:
            print(f"Warning: Metric '{col}' not found in dataframe, skipping")
            continue

        # Rank by metric (higher is better for most confidence metrics)
        rank_df[f"rank_{col}"] = (
            df[col]
            .rank(method="min", ascending=False)  # Higher values = better ranks
            .astype(int)
            / inverse_importance  # Scale by importance
        )

    # Take worst (max) scaled rank as quality key
    df["max_rank"] = rank_df.max(axis=1)

    # Sort by max_rank, create dense final rank
    df = df.sort_values("max_rank")
    df["secondary_rank"] = df["max_rank"].rank(method="dense").astype(int)

    # Tiebreak by iptm
    tiebreak_col = "design_to_target_iptm" if "design_to_target_iptm" in df.columns else "design_iptm"
    if tiebreak_col in df.columns:
        df = df.sort_values(
            by=["secondary_rank", tiebreak_col],
            ascending=[True, False]  # Lower rank, higher iptm
        )

    # Assign final ranks
    df["final_rank"] = np.arange(1, len(df) + 1)

    # Convert to quality score [0, 1]
    if len(df) > 1:
        df["quality_score"] = 1 - (df["final_rank"] - 1) / (len(df) - 1)
    else:
        df["quality_score"] = 1.0

    return df


def main() -> None:
    """Main entry point for standalone PPI scoring."""
    parser = argparse.ArgumentParser(
        prog="boltzgen-score-ppi",
        description="Score protein-protein interactions using BoltzGen methodology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "input",
        type=Path,
        help="Input structure file (.cif or .pdb) or directory containing structures.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to Boltz-2 checkpoint (e.g., boltz2_fold.ckpt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for PPI scores",
    )

    # Ligand/Receptor specification
    parser.add_argument(
        "--ligand-chains",
        type=str,
        required=True,
        help="Comma-separated chain IDs to treat as ligand/binding partner (e.g., 'B,C'). Required.",
    )
    parser.add_argument(
        "--receptor-chains",
        type=str,
        required=True,
        help="Comma-separated chain IDs to treat as receptor (e.g., 'A'). Required.",
    )

    # Metrics configuration
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="Custom metric weights in format 'metric_name=weight' (higher weight = less important). "
        "Example: --metrics design_to_target_iptm=1 design_ptm=1",
    )

    # Optional arguments
    parser.add_argument(
        "--moldir",
        type=str,
        default=None,
        help="Path to molecule directory. Default: downloads from HuggingFace",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Cache directory for model downloads. Default: ~/.cache",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPU devices to use. Default: 1",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for prediction. Default: 1",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader worker processes. Default: 4",
    )
    parser.add_argument(
        "--recycling-steps",
        type=int,
        default=3,
        help="Number of recycling steps. Default: 3",
    )
    parser.add_argument(
        "--use-kernels",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether to use GPU kernels. 'auto' uses kernels if device capability >= 8.0. Default: auto",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary directory with generated metadata files",
    )
    parser.add_argument(
        "--skip-confidence",
        action="store_true",
        help="Skip Boltz-2 confidence prediction (faster, but fewer metrics)",
    )
    parser.add_argument(
        "--with-refolding",
        action="store_true",
        help="Perform structure refolding and compute RMSD metrics (slower, more comprehensive)",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=200,
        help="Number of diffusion sampling steps for refolding. Default: 200",
    )
    parser.add_argument(
        "--diffusion-samples",
        type=int,
        default=5,
        help="Number of structure samples to generate during refolding. Default: 5",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        print(f"ERROR: Input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint file not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    # Parse ligand and receptor chains
    ligand_chains = [c.strip() for c in args.ligand_chains.split(",")]
    receptor_chains = [c.strip() for c in args.receptor_chains.split(",")]

    # Parse metrics configuration
    default_metrics = {
        "design_to_target_iptm": 1,
        "design_ptm": 1,
        "neg_min_design_to_target_pae": 1,
    }

    # Add refolding metrics to defaults if refolding is enabled
    if args.with_refolding:
        default_metrics.update({
            "neg_refold_rmsd": 1,  # Negative because lower RMSD is better
            "neg_refold_rmsd_design": 1,
            "refold_lddt_intra_design": 2,  # Less important than core metrics
        })

    if args.metrics:
        metrics_config = {}
        for metric_spec in args.metrics:
            if "=" not in metric_spec:
                print(f"ERROR: Invalid metric format '{metric_spec}'. Use 'metric_name=weight'", file=sys.stderr)
                sys.exit(1)
            metric_name, weight_str = metric_spec.split("=", 1)
            try:
                metrics_config[metric_name.strip()] = float(weight_str.strip())
            except ValueError:
                print(f"ERROR: Invalid weight value in '{metric_spec}'", file=sys.stderr)
                sys.exit(1)
    else:
        metrics_config = default_metrics

    # Handle moldir
    moldir = args.moldir
    if moldir is None:
        moldir = "huggingface:boltzgen/inference-data:mols.zip"

    # Resolve moldir path if it's a HuggingFace reference
    if moldir.startswith("huggingface:"):
        from boltzgen.cli.boltzgen import get_artifact_path

        # Create minimal args namespace for get_artifact_path
        class Args:
            def __init__(self, cache_dir):
                self.force_download = False
                self.models_token = None
                self.cache = cache_dir

        moldir_args = Args(args.cache)
        moldir = str(get_artifact_path(moldir_args, moldir, repo_type="dataset", verbose=True))

    # Handle use_kernels setting
    use_kernels_arg = args.use_kernels
    if args.use_kernels == "auto" and torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        use_kernels_value = device_capability[0] >= 8
        use_kernels_arg = "true" if use_kernels_value else "false"
        print(
            f"Auto-detected GPU capability: {device_capability}, using kernels: {use_kernels_value}"
        )

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("BoltzGen PPI Scoring")
    print(f"{'=' * 70}")
    print(f"Input:      {args.input}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output:     {args.output}")
    print(f"Ligand:     chains {', '.join(ligand_chains)}")
    print(f"Receptor:   chains {', '.join(receptor_chains)}")
    print(f"Devices:    {args.devices}")
    print(f"Batch size: {args.batch_size}")
    print(f"\nMetrics configuration:")
    for metric, weight in metrics_config.items():
        print(f"  {metric}: {weight}")
    print(f"{'=' * 70}")

    # Prepare structures (generate metadata)
    temp_dir = None
    try:
        prepared_dir = prepare_structures_for_ppi_scoring(
            args.input,
            Path(moldir),
            ligand_chains,
            receptor_chains,
        )

        # Run confidence prediction with Boltz-2
        confidence_dir = None
        if not args.skip_confidence:
            print(f"\n{'=' * 70}")
            print("Running Boltz-2 confidence prediction...")
            print(f"{'=' * 70}\n")

            # Create configuration
            config = create_confidence_prediction_config(
                prepared_dir=prepared_dir,
                checkpoint=args.checkpoint,
                output=args.output,
                devices=args.devices,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                moldir=moldir,
                recycling_steps=args.recycling_steps,
                use_kernels=use_kernels_arg,
            )

            # Run prediction
            import hydra

            task = hydra.utils.instantiate(config)
            task.run(config)

            confidence_dir = prepared_dir / "confidence"
            if not confidence_dir.exists():
                print(f"Warning: Confidence directory not found at {confidence_dir}")
                confidence_dir = None

        # Run refolding with Boltz-2 if requested
        refolding_dir = None
        if args.with_refolding:
            print(f"\n{'=' * 70}")
            print("Running Boltz-2 structure refolding...")
            print(f"{'=' * 70}\n")

            # Create configuration
            config = create_refolding_config(
                prepared_dir=prepared_dir,
                checkpoint=args.checkpoint,
                output=args.output,
                devices=args.devices,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                moldir=moldir,
                recycling_steps=args.recycling_steps,
                sampling_steps=args.sampling_steps,
                diffusion_samples=args.diffusion_samples,
                use_kernels=use_kernels_arg,
            )

            # Run refolding
            import hydra

            task = hydra.utils.instantiate(config)
            task.run(config)

            refolding_dir = prepared_dir / const.folding_dirname
            if not refolding_dir.exists():
                print(f"Warning: Refolding directory not found at {refolding_dir}")
                refolding_dir = None
            else:
                print(f"Refolding results saved to: {refolding_dir}")

        # Compute interface metrics for each structure
        print(f"\n{'=' * 70}")
        print("Computing interface metrics...")
        print(f"{'=' * 70}\n")

        structure_files = list(prepared_dir.glob("*.cif")) + list(prepared_dir.glob("*.pdb"))
        all_metrics = []

        for struct_file in tqdm(structure_files, desc="Processing structures"):
            metadata_file = struct_file.with_suffix(".npz")
            confidence_file = None
            if confidence_dir:
                confidence_file = confidence_dir / f"{struct_file.stem}.npz"

            refolding_file = None
            if refolding_dir:
                refolding_file = refolding_dir / f"{struct_file.stem}.npz"

            try:
                metrics = compute_interface_metrics(
                    struct_file,
                    metadata_file,
                    confidence_file,
                    refolding_file,
                    Path(moldir) if moldir else None,
                )
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error processing {struct_file.name}: {e}")
                import traceback
                traceback.print_exc()

        if not all_metrics:
            print("ERROR: Failed to compute metrics for any structures", file=sys.stderr)
            sys.exit(1)

        # Create DataFrame
        df = pd.DataFrame(all_metrics)

        # Rank designs
        print(f"\n{'=' * 70}")
        print("Ranking designs...")
        print(f"{'=' * 70}\n")

        df = rank_designs(df, metrics_config)

        # Save results
        output_csv = args.output / "ppi_scores.csv"
        df.to_csv(output_csv, index=False, float_format="%.5f")

        print(f"\n{'=' * 70}")
        print("PPI scoring completed successfully!")
        print(f"Results saved to: {output_csv}")
        print(f"{'=' * 70}\n")

        # Print top 10
        if len(df) > 0:
            print("Top 10 designs by quality score:")
            print(df[["id", "final_rank", "quality_score"]].head(10).to_string(index=False))
            print()

        # Cleanup or keep temp directory
        if args.keep_temp:
            print(f"Temporary directory preserved: {prepared_dir}")
        else:
            shutil.rmtree(prepared_dir)

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
