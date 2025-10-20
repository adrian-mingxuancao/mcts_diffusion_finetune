"""
Shared metrics and reward helpers for forward folding tasks.

These utilities provide a single source of truth for:
- Extracting comparable CA coordinates from model outputs
- Computing RMSD and TM-score with the same formulas used in evaluation scripts
- Deriving the composite folding reward that blends TM-score and biophysical terms

Keeping the logic here ensures rollouts and final evaluations report consistent
numbers regardless of where they are computed.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _extract_ca_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Convert raw coordinate tensors to an (L, 3) array of CA positions.
    
    Accepts tensors that may come from different sources:
    - (L, 3) direct CA coordinates
    - (L, 14, 3) or (L, 37, 3) backbone/atom grids
    - (1, L, 14, 3) or (8, 1, L, 14, 3) batched tensors from ESMFold
    """
    if coords is None:
        raise ValueError("Coordinate array is None")
    
    array = np.asarray(coords)
    
    # Handle batched tensors (e.g., [8, 1, L, 14, 3] or [1, L, 14, 3])
    while array.ndim >= 4:
        array = array[0]
    
    if array.ndim == 3:
        # Expect shape (L, atoms, 3)
        if array.shape[-1] != 3:
            raise ValueError(f"Unexpected coordinate shape {array.shape}, expected last dim size 3")
        atom_axis = 1 if array.shape[1] in (14, 37) else 0
        if atom_axis == 1:
            return array[:, 1, :]
        # If axis 0 encodes atoms (rare), fall back to CA index 1 when possible
        if array.shape[0] in (14, 37):
            return array[1, :, :]
        return array
    
    if array.ndim == 2:
        if array.shape[1] != 3:
            raise ValueError(f"2D coordinate array expected shape (L, 3), got {array.shape}")
        return array
    
    raise ValueError(f"Unsupported coordinate tensor shape: {array.shape}")


def _compute_tm_score(distances: np.ndarray, L_target: int) -> float:
    """Compute TM-score using the standard length-dependent scaling."""
    if L_target <= 0:
        return 0.0
    if L_target > 15:
        d_0 = 1.24 * ((L_target - 15) ** (1 / 3)) - 1.8
    else:
        d_0 = 0.5
    denom = 1.0 + (distances / d_0) ** 2
    return float(np.sum(1.0 / denom) / L_target)


def calculate_biophysical_score(sequence: str) -> float:
    """Replicate the CAMEO-style biophysical heuristic used for folding rewards."""
    if not sequence:
        return 0.8
    try:
        seq = sequence.upper()
        length = len(seq)
        positive = sum(1 for aa in seq if aa in "KRH")
        negative = sum(1 for aa in seq if aa in "DE")
        charge_imbalance = abs(positive - negative) / length
        charge_penalty = min(0.3, charge_imbalance * 0.5)
        
        hydrophobic_runs = []
        current_run = 0
        for aa in seq:
            if aa in "AILMFPWV":
                current_run += 1
            else:
                if current_run:
                    hydrophobic_runs.append(current_run)
                current_run = 0
        if current_run:
            hydrophobic_runs.append(current_run)
        max_run = max(hydrophobic_runs) if hydrophobic_runs else 0
        hydrophobic_penalty = 0.0
        if max_run > 3:
            hydrophobic_penalty = min(0.2, (max_run - 3) * 0.05)
        
        base_score = 1.0 - charge_penalty - hydrophobic_penalty
        return float(np.clip(base_score, 0.0, 1.0))
    except Exception:
        return 0.8


def calculate_folding_reward(tm_score: float, sequence: str, aar: float = 1.0) -> float:
    """
    Composite folding reward aligned with evaluation scripts.
    
    Reward weights mirror the sampling and analysis utilities:
        R = 0.4 * AAR + 0.45 * TM-score + 0.15 * biophysical score
    """
    biophysical = calculate_biophysical_score(sequence)
    reward = 0.4 * aar + 0.45 * tm_score + 0.15 * biophysical
    return float(np.clip(reward, 0.0, 1.0))


def evaluate_folding_metrics(
    predicted_coords: np.ndarray,
    reference_coords: np.ndarray,
    sequence: str,
    aar: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Compute RMSD, TM-score, and composite reward for folding.
    
    Returns:
        (rmsd_angstrom, tm_score, composite_reward)
    """
    pred_ca = _extract_ca_coordinates(predicted_coords)
    ref_ca = _extract_ca_coordinates(reference_coords)
    
    min_len = min(len(pred_ca), len(ref_ca))
    if min_len == 0:
        return float("inf"), 0.0, calculate_folding_reward(0.0, sequence or "", aar)
    
    pred_trimmed = pred_ca[:min_len]
    ref_trimmed = ref_ca[:min_len]
    
    diffs = pred_trimmed - ref_trimmed
    rmsd = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    tm_score = _compute_tm_score(distances, min_len)
    reward = calculate_folding_reward(tm_score, sequence or "", aar)
    
    return rmsd, tm_score, reward

