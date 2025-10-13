#!/usr/bin/env python3
"""
Inverse Folding Sampling (Depth=0) - Direct Rollout Benchmark

This script performs sampling-based inverse folding on the CAMEO dataset
without building an MCTS tree. For each structure we:
    1. Load the reference sequence and structure tokens
    2. Obtain a DPLM-2 baseline sequence (cached or freshly generated)
    3. Sample additional sequences from the requested experts
    4. Evaluate AAR, scTM (optional), biophysical score, and composite reward
    5. Save detailed per-structure results and an aggregate summary

By default the script iterates over *all* CAMEO structures so it can be
submitted to Slurm for full-dataset evaluation.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio import SeqIO

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from mcts_diffusion_finetune.core.dplm2_integration import DPLM2Integration  # noqa: E402
from mcts_diffusion_finetune.utils.cameo_data_loader import CAMEODataLoader  # noqa: E402

try:
    from mcts_diffusion_finetune.utils.sctm_calculation import calculate_sctm_score  # noqa: E402
    SCTM_AVAILABLE = True
except Exception:
    SCTM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    """Configure root logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_reference_sequences(fasta_path: Path) -> Dict[str, str]:
    """Load reference AA sequences from aatype.fasta."""
    sequences: Dict[str, str] = {}
    if not fasta_path.exists():
        logging.warning("Reference FASTA not found: %s", fasta_path)
        return sequences

    for record in SeqIO.parse(str(fasta_path), "fasta"):
        sequences[record.id] = str(record.seq).replace(" ", "").upper()
    logging.info("Loaded %d reference sequences", len(sequences))
    return sequences


def load_structure_tokens(fasta_path: Path) -> Dict[str, str]:
    """Load structure token sequences (comma-separated) from struct.fasta."""
    records: Dict[str, str] = {}
    if not fasta_path.exists():
        logging.warning("Structure FASTA not found: %s", fasta_path)
        return records

    for record in SeqIO.parse(str(fasta_path), "fasta"):
        tokens = str(record.seq).replace("\n", "")
        records[record.id] = normalize_struct_sequence(tokens)
    logging.info("Loaded %d structure token sequences", len(records))
    return records


def normalize_struct_sequence(raw_tokens: str) -> str:
    """Normalize structure token sequence string (remove whitespace, trailing commas)."""
    tokens = [tok.strip() for tok in raw_tokens.split(",") if tok.strip()]
    return ",".join(tokens)


def calculate_simple_aar(pred_seq: str, ref_seq: str) -> float:
    """Calculate Amino Acid Recovery (fraction of identical residues)."""
    length = min(len(pred_seq), len(ref_seq))
    if length == 0:
        return 0.0
    matches = sum(1 for a, b in zip(pred_seq[:length], ref_seq[:length]) if a == b)
    return matches / length


def calculate_biophysical_score(sequence: str) -> float:
    """Simple biophysical heuristic (penalize extreme charge/hydrophobic content)."""
    if not sequence:
        return 0.0

    length = len(sequence)
    charged = sum(1 for aa in sequence if aa in "DEKR") / length
    hydrophobic = sum(1 for aa in sequence if aa in "AILMFWYV") / length

    charge_penalty = max(0.0, charged - 0.30) * 2.0
    hydrophobic_penalty = max(0.0, hydrophobic - 0.40) * 2.0

    score = 1.0 - charge_penalty - hydrophobic_penalty
    return max(0.0, min(1.0, score))


def calculate_composite_reward(aar: float, sctm: float, biophysical: float) -> float:
    """Composite reward matching the inverse folding ablation study."""
    return 0.40 * aar + 0.45 * sctm + 0.15 * biophysical


def clean_sequence(seq: str) -> str:
    """Filter sequence to standard 20 amino acids."""
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(aa for aa in seq.upper() if aa in valid)


def load_pregenerated_baseline(structure_id: str) -> Optional[str]:
    """Load pregenerated DPLM-2 baseline sequence for structure (if available)."""
    baseline_dir = Path("/home/caom/AID3/dplm/generation-results/dplm2_150m/inverse_folding")
    fasta_path = baseline_dir / f"{structure_id}.fasta"
    if not fasta_path.exists():
        return None

    try:
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            return clean_sequence(record.seq)
    except Exception as exc:
        logging.warning("Failed to load baseline for %s: %s", structure_id, exc)
    return None


def extract_reference_coords(structure: Dict) -> Optional[np.ndarray]:
    """Extract CA coordinates from a CAMEO structure dictionary."""
    candidates = [
        structure.get("coordinates"),
        structure.get("atom_positions"),
        structure.get("backbone_coords"),
    ]
    for coords in candidates:
        if coords is None:
            continue
        arr = np.asarray(coords)
        if arr.ndim == 3 and arr.shape[1] >= 2:
            # DPLM convention: [:, 1, :] == CA atoms
            return arr[:, 1, :]
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
    return None


def prepare_structure_record(
    structure_id: str,
    struct_tokens: str,
    cameo_structure: Optional[Dict],
) -> Dict:
    """Create baseline structure dictionary expected by DPLM-2 integration."""
    pdb_id, chain_id = (structure_id.split("_") + ["A"])[:2]

    record = {
        "name": f"CAMEO {structure_id}",
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "struct_seq": struct_tokens,
        "length": len(struct_tokens.split(",")),
    }

    if cameo_structure:
        for key in [
            "coordinates",
            "atom_positions",
            "backbone_coords",
            "struct_ids",
            "plddt_scores",
        ]:
            if key in cameo_structure and cameo_structure[key] is not None:
                record[key] = cameo_structure[key]
    return record


def sample_sequence(
    dplm2: DPLM2Integration,
    struct_tokens: str,
    target_length: int,
    expert_id: int,
    temperature: float,
) -> Optional[str]:
    """Sample a sequence from DPLM-2 given structure tokens."""
    masked_aa = dplm2.tokenizer.aa_mask_token * target_length
    return dplm2.generate_from_masked_input(
        aa_sequence=masked_aa,
        struct_tokens=struct_tokens,
        task_type="inverse_folding",
        expert_id=expert_id,
        temperature=temperature,
    )


def resolve_expert_ids(expert_names: List[str]) -> List[int]:
    """Map expert name strings to DPLM-2 expert indices."""
    mapping = {
        "dplm2_650m": 0,
        "dplm2_150m": 1,
        "dplm2_3b": 2,
    }
    expert_ids: List[int] = []
    for name in expert_names:
        key = name.lower()
        if key not in mapping:
            logging.warning("Unknown expert '%s' – skipping", name)
            continue
        expert_ids.append(mapping[key])
    if not expert_ids:
        logging.info("No valid experts specified, defaulting to DPLM-2 150M")
        expert_ids = [1]
    return expert_ids


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sampling-only inverse folding benchmark on CAMEO."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/caom/AID3/dplm/data-bin/cameo2022",
        help="Path to CAMEO 2022 dataset root.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index within the sorted CAMEO structure list (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index within the structure list (exclusive). Defaults to all structures.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Samples per expert for each structure.",
    )
    parser.add_argument(
        "--experts",
        type=str,
        default="dplm2_650m,dplm2_150m,dplm2_3b",
        help="Comma-separated list of experts (dplm2_650m,dplm2_150m,dplm2_3b).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature passed to DPLM-2.",
    )
    parser.add_argument(
        "--skip_sctm",
        action="store_true",
        help="Skip scTM calculation (saves time and avoids ESMFold dependency).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/net/scratch/caom/cameo_evaluation_results",
        help="Directory for JSON results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Base random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_path)
    reference_sequences = load_reference_sequences(data_root / "aatype.fasta")
    struct_sequences = load_structure_tokens(data_root / "struct.fasta")

    loader = CAMEODataLoader(str(data_root))
    structure_ids = [Path(s).stem for s in loader.structures]
    total_structures = len(structure_ids)
    if total_structures == 0:
        logging.error("No CAMEO structures found under %s", data_root)
        return

    start_idx = max(0, args.start)
    end_idx = args.end if args.end is not None else total_structures
    end_idx = min(end_idx, total_structures)
    if start_idx >= end_idx:
        logging.error("Invalid range: start=%d end=%d (dataset size=%d)", start_idx, end_idx, total_structures)
        return

    selected_ids = structure_ids[start_idx:end_idx]
    logging.info(
        "Processing %d structures (indices %d:%d)",
        len(selected_ids),
        start_idx,
        end_idx,
    )

    expert_ids = resolve_expert_ids([name.strip() for name in args.experts.split(",") if name.strip()])
    logging.info("Experts: %s", expert_ids)

    compute_sctm = SCTM_AVAILABLE and not args.skip_sctm
    if not SCTM_AVAILABLE and not args.skip_sctm:
        logging.warning("scTM computation unavailable (ESMFold/TMalign missing). Using fallback scores.")

    dplm2 = DPLM2Integration(device="cuda", default_temperature=args.temperature)

    results: List[Dict] = []
    best_rewards: List[float] = []
    best_aars: List[float] = []
    best_sctm: List[float] = []

    for idx, structure_id in enumerate(selected_ids, start=start_idx):
        logging.info("(%d/%d) Structure %s", idx + 1, end_idx, structure_id)

        ref_seq = reference_sequences.get(structure_id)
        if not ref_seq:
            logging.warning("Missing reference sequence for %s – skipping", structure_id)
            continue

        struct_tokens = struct_sequences.get(structure_id)
        if not struct_tokens:
            logging.warning("Missing structure tokens for %s – skipping", structure_id)
            continue

        cameo_struct = loader.load_structure(structure_id)
        structure_record = prepare_structure_record(structure_id, struct_tokens, cameo_struct)
        reference_coords = extract_reference_coords(cameo_struct) if cameo_struct else None

        baseline_seq = load_pregenerated_baseline(structure_id)
        if not baseline_seq:
            logging.info("Generating baseline with DPLM-2 150M...")
            baseline_seq = sample_sequence(
                dplm2,
                struct_tokens=struct_tokens,
                target_length=len(ref_seq),
                expert_id=1,
                temperature=args.temperature,
            )
        if not baseline_seq:
            logging.warning("Baseline generation failed for %s – skipping", structure_id)
            continue

        baseline_seq = clean_sequence(baseline_seq)
        baseline_aar = calculate_simple_aar(baseline_seq, ref_seq)
        baseline_bio = calculate_biophysical_score(baseline_seq)
        baseline_sctm = 0.5
        if compute_sctm and reference_coords is not None:
            try:
                baseline_sctm = float(calculate_sctm_score(baseline_seq, reference_coords))
            except Exception as exc:
                logging.warning("scTM baseline failure (%s): %s", structure_id, exc)
        baseline_reward = calculate_composite_reward(baseline_aar, baseline_sctm, baseline_bio)

        samples: List[Dict] = []
        best_sample = {
            "sequence": baseline_seq,
            "aar": baseline_aar,
            "sctm": baseline_sctm,
            "biophysical": baseline_bio,
            "reward": baseline_reward,
            "expert_id": "baseline",
            "sample_index": -1,
        }

        sample_counter = 0
        for expert_id in expert_ids:
            for sample_idx in range(args.num_samples):
                sample_counter += 1
                seed = args.seed + sample_counter + int(time.time()) % 1000
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                try:
                    import torch

                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                except Exception:
                    pass

                candidate = sample_sequence(
                    dplm2,
                    struct_tokens=struct_tokens,
                    target_length=len(ref_seq),
                    expert_id=expert_id,
                    temperature=args.temperature,
                )
                if not candidate:
                    logging.warning("Sample failed (expert %d, idx %d)", expert_id, sample_idx + 1)
                    continue

                candidate = clean_sequence(candidate)
                if len(candidate) != len(ref_seq):
                    logging.warning(
                        "Length mismatch for %s (expert %d sample %d): %d vs %d",
                        structure_id,
                        expert_id,
                        sample_idx + 1,
                        len(candidate),
                        len(ref_seq),
                    )
                    continue

                aar = calculate_simple_aar(candidate, ref_seq)
                bio = calculate_biophysical_score(candidate)
                sctm = baseline_sctm  # default fallback
                if compute_sctm and reference_coords is not None:
                    try:
                        sctm = float(calculate_sctm_score(candidate, reference_coords))
                    except Exception as exc:
                        logging.warning("scTM failure (%s expert %d sample %d): %s", structure_id, expert_id, sample_idx + 1, exc)

                reward = calculate_composite_reward(aar, sctm, bio)
                sample_record = {
                    "sequence": candidate,
                    "aar": aar,
                    "sctm": sctm,
                    "biophysical": bio,
                    "reward": reward,
                    "expert_id": expert_id,
                    "sample_index": sample_idx + 1,
                }
                samples.append(sample_record)

                if reward > best_sample["reward"]:
                    best_sample = sample_record

        record = {
            "structure_id": structure_id,
            "reference_length": len(ref_seq),
            "baseline": {
                "sequence": baseline_seq,
                "aar": baseline_aar,
                "sctm": baseline_sctm,
                "biophysical": baseline_bio,
                "reward": baseline_reward,
            },
            "best_sample": best_sample,
            "samples": samples,
        }
        results.append(record)
        best_rewards.append(best_sample["reward"])
        best_aars.append(best_sample["aar"])
        best_sctm.append(best_sample["sctm"])

        logging.info(
            "Best reward %.3f (AAR %.3f, scTM %.3f, Expert %s)",
            best_sample["reward"],
            best_sample["aar"],
            best_sample["sctm"],
            best_sample["expert_id"],
        )

    if not results:
        logging.warning("No successful structures processed.")
        return

    mean_reward = float(np.mean(best_rewards))
    mean_aar = float(np.mean(best_aars))
    mean_sctm = float(np.mean(best_sctm))
    logging.info("==== Sampling Summary ====")
    logging.info("Structures processed: %d", len(results))
    logging.info("Mean best reward: %.3f", mean_reward)
    logging.info("Mean best AAR   : %.3f", mean_aar)
    logging.info("Mean best scTM  : %.3f", mean_sctm)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"inverse_folding_sampling_{timestamp}.json"
    summary_path = output_dir / f"inverse_folding_sampling_summary_{timestamp}.json"

    with result_path.open("w") as fh:
        json.dump(results, fh, indent=2)

    summary_payload = {
        "timestamp": timestamp,
        "num_structures": len(results),
        "mean_best_reward": mean_reward,
        "mean_best_aar": mean_aar,
        "mean_best_sctm": mean_sctm,
        "experts": expert_ids,
        "num_samples_per_expert": args.num_samples,
        "temperature": args.temperature,
        "skip_sctm": not compute_sctm,
        "start_index": start_idx,
        "end_index": end_idx,
        "result_file": str(result_path),
    }
    with summary_path.open("w") as fh:
        json.dump(summary_payload, fh, indent=2)

    logging.info("Saved detailed results to %s", result_path)
    logging.info("Saved run summary to %s", summary_path)


if __name__ == "__main__":
    main()
