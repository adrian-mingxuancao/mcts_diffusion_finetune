"""
Shared metrics and reward helpers for forward folding tasks.

These utilities provide a single source of truth for:
- Extracting comparable CA coordinates from model outputs
- Computing RMSD and TM-score with the same formulas used in evaluation scripts
- Deriving the composite folding reward that blends TM-score and biophysical terms

Keeping the logic here ensures rollouts and final evaluations report consistent
numbers regardless of where they are computed.

IMPORTANT: For folding tasks, we use:
- RMSD with optimal superimposition (Kabsch algorithm via OpenFold)
- TM-score with optimal alignment (tm_align from tmtools)
- Reward based ONLY on TM-score (no AAR since sequence is fixed)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


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


def calculate_folding_reward(
    tm_score: float, 
    rmsd: float = None,
    plddt: float = None,
    sequence: str = "", 
    aar: float = 1.0
) -> float:
    """
    Folding reward combining TM-score, RMSD, and pLDDT.
    
    Formula: R_fold = α·TM + β·(1 - min(RMSD/10, 1)) + γ·pLDDT
    
    Default weights (if all components available):
        α = 0.4 (TM-score weight)
        β = 0.3 (RMSD weight)
        γ = 0.3 (pLDDT weight)
    
    If pLDDT not available: α=0.6, β=0.4, γ=0
    If RMSD not available: α=0.7, β=0, γ=0.3
    If only TM available: α=1.0, β=0, γ=0
    
    For FOLDING tasks:
        - Sequence is FIXED (given as input)
        - AAR is always 1.0 (not being optimized)
        - Reward based on structure quality (TM, RMSD, pLDDT)
    """
    # TM-score component (always available)
    tm_component = float(np.clip(tm_score, 0.0, 1.0))
    
    # RMSD component (lower is better, normalize to 0-1 range)
    rmsd_component = 0.0
    if rmsd is not None and rmsd != float('inf'):
        # 1 - min(RMSD/10, 1): RMSD=0 gives 1.0, RMSD>=10 gives 0.0
        rmsd_component = 1.0 - min(rmsd / 10.0, 1.0)
    
    # pLDDT component (already in 0-100 range, normalize to 0-1)
    plddt_component = 0.0
    if plddt is not None:
        plddt_component = float(np.clip(plddt / 100.0, 0.0, 1.0))
    
    # Adaptive weighting based on available components
    if rmsd is not None and plddt is not None:
        # All components available
        alpha, beta, gamma = 0.4, 0.3, 0.3
        reward = alpha * tm_component + beta * rmsd_component + gamma * plddt_component
    elif rmsd is not None:
        # TM and RMSD available
        alpha, beta = 0.6, 0.4
        reward = alpha * tm_component + beta * rmsd_component
    elif plddt is not None:
        # TM and pLDDT available
        alpha, gamma = 0.7, 0.3
        reward = alpha * tm_component + gamma * plddt_component
    else:
        # Only TM available (fallback)
        reward = tm_component
    
    return float(np.clip(reward, 0.0, 1.0))


def evaluate_folding_metrics(
    predicted_coords: np.ndarray,
    reference_coords: np.ndarray,
    sequence: str,
    aar: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Compute RMSD, TM-score, and composite reward for folding.
    
    Uses official evaluation methods:
    - RMSD: Optimal superimposition (Kabsch algorithm)
    - TM-score: Optimal alignment (tm_align from tmtools)
    
    Returns:
        (rmsd_angstrom, tm_score, composite_reward)
    """
    pred_ca = _extract_ca_coordinates(predicted_coords)
    ref_ca = _extract_ca_coordinates(reference_coords)
    
    min_len = min(len(pred_ca), len(ref_ca))
    if min_len == 0:
        return float("inf"), 0.0, calculate_folding_reward(0.0, rmsd=None, plddt=None)
    
    pred_trimmed = pred_ca[:min_len]
    ref_trimmed = ref_ca[:min_len]
    
    try:
        # Use OpenFold's superimpose for RMSD (with optimal alignment)
        from openfold.utils.superimposition import superimpose
        
        pred_tensor = torch.tensor(pred_trimmed, dtype=torch.float32)[None]  # [1, L, 3]
        ref_tensor = torch.tensor(ref_trimmed, dtype=torch.float32)[None]    # [1, L, 3]
        mask = torch.ones(min_len, dtype=torch.bool)
        
        _, rmsd_tensor = superimpose(pred_tensor, ref_tensor, mask)
        rmsd = float(rmsd_tensor.item())
    except Exception as e:
        print(f"   ⚠️ Superimpose failed, using unaligned RMSD: {e}")
        # Fallback to unaligned RMSD
        diffs = pred_trimmed - ref_trimmed
        rmsd = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))
    
    try:
        # Use tmtools for TM-score (with optimal alignment)
        from tmtools import tm_align
        
        # tm_align expects (L, 3, 3) for backbone atoms, but we only have CA
        # Create dummy backbone with CA at position 1
        pred_bb = np.zeros((min_len, 3, 3), dtype=np.float64)
        ref_bb = np.zeros((min_len, 3, 3), dtype=np.float64)
        pred_bb[:, 1, :] = pred_trimmed  # CA at index 1
        ref_bb[:, 1, :] = ref_trimmed
        
        # Create dummy sequence (tm_align needs it but we only care about structure)
        dummy_seq = "A" * min_len
        
        tm_results = tm_align(pred_bb, ref_bb, dummy_seq, dummy_seq)
        tm_score = float(tm_results.tm_norm_chain1)  # TM-score normalized by chain 1 length
    except Exception as e:
        print(f"   ⚠️ tm_align failed, using simple TM-score: {e}")
        # Fallback to simple TM-score calculation
        distances = np.sqrt(np.sum((pred_trimmed - ref_trimmed) ** 2, axis=1))
        tm_score = _compute_tm_score(distances, min_len)
    
    # Calculate composite reward with TM-score and RMSD
    # Note: pLDDT not available from this evaluator (would need ESMFold output)
    reward = calculate_folding_reward(tm_score, rmsd=rmsd, plddt=None)
    
    return rmsd, tm_score, reward

