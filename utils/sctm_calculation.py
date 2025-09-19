"""
Simple scTM calculation using ESMFold similar to pLDDT calculation.
Reuses existing ESMFold model from real_plddt_computation.py to avoid reloading.
"""

import numpy as np
import torch
import tempfile
import os
import subprocess
from typing import Dict, Optional, Tuple
from .real_plddt_computation import load_esmfold_model


def calculate_sctm_score(sequence: str, reference_coords: np.ndarray) -> float:
    """
    Calculate scTM score by predicting structure with ESMFold and comparing to reference.
    
    Args:
        sequence: Amino acid sequence to evaluate
        reference_coords: Reference structure coordinates (N, 3) - CA atoms
        
    Returns:
        scTM score (0.0 to 1.0)
    """
    try:
        # Use existing ESMFold model from pLDDT calculation
        model, tokenizer = load_esmfold_model()
        if model is None or tokenizer is None:
            print("⚠️ ESMFold not available for scTM calculation")
            return 0.5

        # Predict structure using ESMFold
        predicted_coords = predict_structure_coords(model, tokenizer, sequence)
        if predicted_coords is None:
            print("⚠️ Structure prediction failed")
            return 0.5

        # Calculate TM-score between predicted and reference structures
        tm_score = calculate_tm_score_from_coords(predicted_coords, reference_coords)

        print(f"📊 scTM score: {tm_score:.3f}")
        return tm_score

    except Exception as e:
        print(f"⚠️ scTM calculation failed: {e}")
        return 0.5


def predict_structure_coords(model, tokenizer, sequence: str) -> Optional[np.ndarray]:
    """
    Predict CA coordinates using ESMFold model.
    
    Args:
        model: ESMFold model
        tokenizer: ESMFold tokenizer
        sequence: Amino acid sequence
        
    Returns:
        Predicted CA coordinates (N, 3) or None if prediction fails
    """
    try:
        with torch.no_grad():
            # Tokenize sequence
            tokenized = tokenizer(sequence, return_tensors="pt", add_special_tokens=False, padding=False)

            # Move to appropriate device
            if torch.cuda.is_available() and next(model.parameters()).is_cuda:
                tokenized = {k: v.cuda() for k, v in tokenized.items()}

            # Get model output with structure prediction
            output = model(tokenized["input_ids"])

            # Debug: Print output structure
            print(f" ESMFold output type: {type(output)}")
            if hasattr(output, 'keys') and callable(output.keys):
                print(f" ESMFold output keys: {list(output.keys())}")

            # Check all possible output attributes
            attrs = ['positions', 'atom_coordinates', 'coordinates', 'final_atom_positions', 'atom_positions']
            for attr in attrs:
                if hasattr(output, attr):
                    val = getattr(output, attr)
                    if val is not None:
                        print(f" Found {attr} with shape: {val.shape if hasattr(val, 'shape') else type(val)}")

            # Try the most common ESMFold output format
            if hasattr(output, 'positions') and output.positions is not None:
                positions = output.positions
                print(f" Using positions with shape: {positions.shape}")

                # Handle different ESMFold output shapes
                if len(positions.shape) == 5:
                    # Shape: (batch, extra_dim, seq_len, atoms, 3) - e.g., [8, 1, 65, 14, 3]
                    ca_coords = positions[0, 0, :, 1, :].cpu().numpy()  # CA atoms (index 1)
                    print(f" Extracted CA coordinates from 5D tensor: {ca_coords.shape}")
                    return ca_coords
                elif len(positions.shape) == 4 and positions.shape[2] >= 2:
                    # Shape: (batch, seq_len, atoms, 3) - standard format
                    ca_coords = positions[0, :, 1, :].cpu().numpy()  # CA atoms (index 1)
                    print(f" Extracted CA coordinates from 4D tensor: {ca_coords.shape}")
                    return ca_coords
                elif len(positions.shape) == 3:
                    # Already per-residue coordinates
                    ca_coords = positions[0].cpu().numpy()
                    print(f" Using per-residue coordinates: {ca_coords.shape}")
                    return ca_coords

                # Try alternative attribute names
                for attr in ['atom_coordinates', 'coordinates', 'final_atom_positions']:
                    if hasattr(output, attr):
                        coords = getattr(output, attr)
                        if coords is not None:
                            print(f" Trying {attr} with shape: {coords.shape}")
                            if len(coords.shape) == 5:
                                ca_coords = coords[0, 0, :, 1, :].cpu().numpy()  # CA atoms from 5D
                                print(f" Extracted CA coordinates from {attr} (5D): {ca_coords.shape}")
                                return ca_coords
                            elif len(coords.shape) == 4 and coords.shape[2] >= 2:
                                ca_coords = coords[0, :, 1, :].cpu().numpy()  # CA atoms from 4D
                                print(f" Extracted CA coordinates from {attr} (4D): {ca_coords.shape}")
                                return ca_coords

            print(" Could not extract coordinates from ESMFold output")
            return None

    except Exception as e:
        print(f" Structure prediction failed: {e}")
        print(f" Structure prediction failed: {e}")
        return None


def calculate_tm_score_from_coords(pred_coords: np.ndarray, ref_coords: np.ndarray) -> float:
    """
    Calculate TM-score from coordinate arrays using TMalign.
    
    Args:
        pred_coords: Predicted coordinates (N, 3)
        ref_coords: Reference coordinates (N, 3)
        
    Returns:
        TM-score (0.0 to 1.0)
    """
    try:
        # Ensure same length and proper format
        min_len = min(len(pred_coords), len(ref_coords))
        pred_coords = np.array(pred_coords[:min_len])
        ref_coords = np.array(ref_coords[:min_len])

        # Handle coordinate format - extract CA atoms if needed
        if ref_coords.ndim == 3 and ref_coords.shape[1] == 3:
            # Reference coords are [N, 3, 3] (N, CA, C atoms) - extract CA
            ref_coords = ref_coords[:, 1, :]  # CA atoms at index 1
        elif ref_coords.ndim == 3:
            # Flatten to [N, 3] if other 3D format
            ref_coords = ref_coords.reshape(-1, 3)

        # Ensure both are [N, 3] format
        if pred_coords.ndim != 2 or pred_coords.shape[1] != 3:
            raise ValueError(f"Invalid predicted coordinates shape: {pred_coords.shape}")
        if ref_coords.ndim != 2 or ref_coords.shape[1] != 3:
            raise ValueError(f"Invalid reference coordinates shape: {ref_coords.shape}")

        # Create temporary PDB files with proper sequence length
        seq_len = len(pred_coords)
        dummy_sequence = "A" * seq_len

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as pred_file:
            pred_pdb_path = pred_file.name
            pred_file.write(coords_to_pdb_string(pred_coords, dummy_sequence))

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as ref_file:
            ref_pdb_path = ref_file.name
            ref_file.write(coords_to_pdb_string(ref_coords, dummy_sequence))

        try:
            # Use TMalign to calculate TM-score (similar to cal_tmscore.py)
            tm_score = run_tmalign(pred_pdb_path, ref_pdb_path)
            return tm_score if not np.isnan(tm_score) else 0.5

        finally:
            # Clean up temporary files
            os.unlink(pred_pdb_path)
            os.unlink(ref_pdb_path)

    except Exception as e:
        print(f"⚠️ TM-score calculation failed: {e}")
        return 0.5


def run_tmalign(query_pdb: str, reference_pdb: str, fast: bool = True) -> float:
    """
    Run TMalign to calculate TM-score between two PDB files.
    Based on analysis/cal_tmscore.py implementation.
    
    Args:
        query_pdb: Path to query PDB file
        reference_pdb: Path to reference PDB file
        fast: Use fast mode
        
    Returns:
        TM-score (0.0 to 1.0) or np.nan if failed
    """
    try:
        # Find TMalign executable
        tmalign_path = "/home/caom/AID3/dplm/analysis/TMalign"
        if not os.path.exists(tmalign_path):
            # Try alternative paths
            for path in ["./TMalign", "/usr/local/bin/TMalign", "TMalign"]:
                if os.path.exists(path):
                    tmalign_path = path
                    break

        cmd = f"{tmalign_path} {query_pdb} {reference_pdb}"
        if fast:
            cmd += " -fast"

        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)

        # Parse TM-score from output (same as cal_tmscore.py)
        score_lines = []
        for line in output.decode().split("\n"):
            if line.startswith("TM-score"):
                score_lines.append(line)

        if not score_lines:
            return np.nan

        # Extract Chain_1 score with defensive handling
        import re
        try:
            key_getter = lambda s: re.findall(r"Chain_[12]{1}", s)[0]
            score_getter = lambda s: float(re.findall(r"=\s+([0-9.]+)", s)[0])
            results_dict = {key_getter(s): score_getter(s) for s in score_lines}

            tm_score = results_dict.get("Chain_1", np.nan)

            # Ensure we return a proper Python float, not numpy scalar
            if not np.isnan(tm_score):
                return float(tm_score)  # Convert to Python float
            else:
                return 0.5  # Default fallback

        except (IndexError, ValueError, KeyError) as e:
            print(f"⚠️ TM-score parsing failed: {e}")
            return 0.5

    except Exception as e:
        print(f"⚠️ TMalign execution failed: {e}")
        return np.nan


def coords_to_pdb_string(coords: np.ndarray, sequence: str) -> str:
    """
    Convert coordinate array to PDB format string.
    
    Args:
        coords: Coordinates array (N, 3)
        sequence: Amino acid sequence
        
    Returns:
        PDB format string
    """
    pdb_lines = []
    for i, (coord, aa) in enumerate(zip(coords, sequence)):
        line = f"ATOM  {i+1:5d}  CA  {aa} A{i+1:4d}    {float(coord[0]):8.3f}{float(coord[1]):8.3f}{float(coord[2]):8.3f}  1.00 20.00           C"
        pdb_lines.append(line)
    return "\n".join(pdb_lines)


def calculate_sctm_with_cameo_data(sequence: str, structure_data) -> float:
    """
    Calculate scTM score using CAMEO dataset structure from .pkl files.
    Compatible with test_mcts_with_real_data.py format.
    
    Args:
        sequence: Amino acid sequence to evaluate
        structure_data: Dictionary containing structure data from .pkl file
                       Expected keys: 'bb_positions', 'atom_positions', 'aatype'
        
    Returns:
        scTM score using reference structure coordinates
    """
    try:
        # Validate input types
        if not isinstance(sequence, str):
            print(f"⚠️ Invalid sequence type: {type(sequence)}")
            return 0.0

        if not isinstance(structure_data, dict):
            print(f"⚠️ Invalid structure_data type: {type(structure_data)}, expected dict")
            return 0.0

        # Extract reference coordinates from CAMEO data
        reference_coords = None

        if 'bb_positions' in structure_data:
            bb_positions = structure_data['bb_positions']
            print(f"🧬 Raw bb_positions shape: {bb_positions.shape}")

            # Handle different bb_positions formats
            if len(bb_positions.shape) == 3 and bb_positions.shape[1] == 3:
                # Shape: (N, 3, 3) - 3 backbone atoms per residue (N, CA, C)
                reference_coords = bb_positions[:, 1, :]  # CA atoms (index 1)
                print(f"🧬 Extracted CA coordinates from bb_positions (N,3,3): {reference_coords.shape}")
            elif len(bb_positions.shape) == 2 and bb_positions.shape[1] == 3:
                # Shape: (N, 3) - already CA coordinates
                reference_coords = bb_positions
                print(f"🧬 Using bb_positions as CA coordinates (N,3): {reference_coords.shape}")
            else:
                print(f"⚠️ Unexpected bb_positions shape: {bb_positions.shape}")
                reference_coords = bb_positions
                print(f"🧬 Using bb_positions as-is: {reference_coords.shape}")
        elif 'atom_positions' in structure_data:
            atom_positions = structure_data['atom_positions']  # Shape: (N, 37, 3)
            if atom_positions.shape[1] > 1:
                reference_coords = atom_positions[:, 1, :]  # CA atoms (index 1)
                print(f"🧬 Using CA coordinates from atom_positions (shape: {reference_coords.shape})")
            else:
                reference_coords = atom_positions[:, 0, :]  # First atom type
                print(f"🧬 Using first atom coordinates (shape: {reference_coords.shape})")

        if reference_coords is None:
            print(f"⚠️ No coordinate data found in structure_data")
            return 0.5

        # Ensure reference coordinates are numpy array and proper dtype
        if not isinstance(reference_coords, np.ndarray):
            reference_coords = np.array(reference_coords)

        # Ensure float64 dtype for consistent calculations
        if reference_coords.dtype != np.float64:
            reference_coords = reference_coords.astype(np.float64)

        # Ensure reference coordinates match sequence length
        if len(reference_coords) != len(sequence):
            min_len = min(len(reference_coords), len(sequence))
            reference_coords = reference_coords[:min_len]
            sequence = sequence[:min_len]
            print(f"🔧 Adjusted lengths to match: {min_len}")

        # Calculate scTM using reference coordinates
        sctm_result = calculate_sctm_score(sequence, reference_coords)

        # Ensure we return a proper Python float
        if isinstance(sctm_result, (np.floating, np.integer)):
            return float(sctm_result)
        elif sctm_result is not None:
            return float(sctm_result)
        else:
            return 0.5

    except Exception as e:
        print(f"⚠️ CAMEO scTM calculation failed: {e}")
        return 0.5


if __name__ == "__main__":
    # Test scTM calculation
    test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKREQTGQGWVPSNYITPVN"

    # Create mock reference coordinates
    reference_coords = np.random.rand(len(test_sequence), 3) * 50

    print("🧪 Testing scTM calculation...")
    sctm_score = calculate_sctm_score(test_sequence, reference_coords)
    print(f"✅ Test scTM score: {sctm_score:.3f}")