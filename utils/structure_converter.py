#!/usr/bin/env python3

"""
Structure token to coordinate conversion utility
Based on the official DPLM-2 evaluator pipeline
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


class StructureTokenConverter:
    """Convert DPLM-2 structure tokens to coordinates using the official pipeline."""

    MASK_TOKENS = {"<mask_struct>", "<mask>", "[mask_struct]"}

    def __init__(
        self,
        struct_tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self._struct_tokenizer = None
        self._struct_tokenizer_path = struct_tokenizer_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._struct_mask_id: Optional[int] = None
        self._max_token_id: Optional[int] = None

    @property
    def struct_tokenizer(self):
        """Load and cache the official struct tokenizer."""
        if self._struct_tokenizer is None:
            try:
                from byprot.models.utils import get_struct_tokenizer

                model_name = (
                    self._struct_tokenizer_path
                    if self._struct_tokenizer_path
                    else "airkingbd/struct_tokenizer"
                )
                print("🔧 Loading structure tokenizer...")
                self._struct_tokenizer = (
                    get_struct_tokenizer(model_name).to(self.device).eval()
                )
                self._initialize_token_metadata()
                print("✅ Structure tokenizer ready")
            except Exception as exc:
                print(f"⚠️ Structure tokenizer loading failed: {exc}")
                self._struct_tokenizer = None
        return self._struct_tokenizer

    def _initialize_token_metadata(self) -> None:
        """Capture mask token id and vocabulary size hints."""
        if self._struct_tokenizer is None:
            return
        mask_id = getattr(self._struct_tokenizer, "struct_mask_id", None)
        quantize = getattr(self._struct_tokenizer, "quantize", None)
        if quantize is not None and hasattr(quantize, "n_embed"):
            self._max_token_id = int(quantize.n_embed) - 1
            if mask_id is None:
                mask_id = self._max_token_id
        self._struct_mask_id = mask_id

    def _flatten_tokens(
        self, structure_tokens: Union[str, List[int], np.ndarray, torch.Tensor]
    ) -> List:
        if structure_tokens is None:
            return []
        if isinstance(structure_tokens, torch.Tensor):
            return torch.flatten(structure_tokens).tolist()
        if isinstance(structure_tokens, np.ndarray):
            return structure_tokens.reshape(-1).tolist()
        if isinstance(structure_tokens, str):
            return [
                token
                for token in re.split(r"[\s,]+", structure_tokens.strip())
                if token
            ]
        if isinstance(structure_tokens, (list, tuple, set)):
            return list(structure_tokens)
        try:
            return list(structure_tokens)
        except TypeError:
            return [structure_tokens]

    def _parse_token_list(
        self, flat_tokens: List, sequence_length: Optional[int]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not flat_tokens:
            return None

        mask_id = (
            self._struct_mask_id
            if self._struct_mask_id is not None
            else (self._max_token_id if self._max_token_id is not None else 0)
        )
        token_ids: List[int] = []
        masked_positions: List[int] = []

        for raw in flat_tokens:
            is_mask = False
            value: Optional[int] = None

            if isinstance(raw, (int, np.integer)):
                value = int(raw)
            elif isinstance(raw, float) and raw.is_integer():
                value = int(raw)
            else:
                token_str = str(raw).strip()
                if not token_str:
                    continue
                lower = token_str.lower()
                if token_str.isdigit():
                    value = int(token_str)
                elif lower in self.MASK_TOKENS:
                    value = mask_id
                    is_mask = True
                else:
                    continue

            if value is None:
                continue

            if self._max_token_id is not None:
                value = max(0, min(value, self._max_token_id))

            token_ids.append(value)
            if is_mask:
                masked_positions.append(len(token_ids) - 1)

        if not token_ids:
            return None

        if (
            sequence_length is not None
            and len(token_ids) == sequence_length + 2
            and len(token_ids) > 2
        ):
            token_ids = token_ids[1:-1]
            masked_positions = [
                pos - 1 for pos in masked_positions if 0 < pos < len(token_ids) + 1
            ]

        if sequence_length is not None and len(token_ids) != sequence_length:
            print(
                f"⚠️ Structure token length mismatch: {len(token_ids)} vs expected {sequence_length}"
            )
            if len(token_ids) > sequence_length:
                token_ids = token_ids[:sequence_length]
                masked_positions = [
                    pos for pos in masked_positions if pos < sequence_length
                ]
            else:
                pad_value = mask_id
                while len(token_ids) < sequence_length:
                    token_ids.append(pad_value)
                    masked_positions.append(len(token_ids) - 1)

        structok = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        res_mask = torch.ones_like(structok, dtype=torch.float32)
        if masked_positions:
            res_mask[0, masked_positions] = 0.0
        return structok, res_mask

    def _prepare_detokenize_inputs(
        self,
        structure_tokens: Union[str, List[int], np.ndarray, torch.Tensor],
        sequence_length: Optional[int] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        flat_tokens = self._flatten_tokens(structure_tokens)
        return self._parse_token_list(flat_tokens, sequence_length)

    def _is_coordinate_array(self, value: Union[np.ndarray, torch.Tensor]) -> bool:
        if isinstance(value, np.ndarray):
            return (
                value.ndim >= 2
                and value.shape[-1] == 3
                and np.issubdtype(value.dtype, np.floating)
            )
        if isinstance(value, torch.Tensor):
            return (
                value.ndim >= 2
                and value.shape[-1] == 3
                and value.dtype.is_floating_point
            )
        return False

    def detokenize_tokens(
        self,
        structure_tokens: Union[str, List[int], np.ndarray, torch.Tensor],
        sequence_length: Optional[int] = None,
    ) -> Optional[dict]:
        tokenizer = self.struct_tokenizer
        if tokenizer is None:
            return None

        inputs = self._prepare_detokenize_inputs(structure_tokens, sequence_length)
        if inputs is None:
            print("⚠️ Unable to prepare structure tokens for detokenization")
            return None

        structok, res_mask = inputs

        try:
            with torch.no_grad():
                decoder_out = tokenizer.detokenize(structok, res_mask)
        except Exception as exc:
            print(f"❌ Detokenization failed: {exc}")
            return None

        seq_len = structok.shape[1]
        device = structok.device

        if (
            "residue_index" not in decoder_out
            or decoder_out["residue_index"] is None
        ):
            decoder_out["residue_index"] = torch.arange(
                seq_len, device=device
            ).long().unsqueeze(0)

        if "chain_index" not in decoder_out or decoder_out["chain_index"] is None:
            decoder_out["chain_index"] = torch.zeros_like(
                decoder_out["residue_index"]
            )

        decoder_out["res_mask"] = res_mask
        return decoder_out

    def tokens_to_coordinates(
        self,
        structure_tokens: Union[str, List[int], np.ndarray, torch.Tensor],
        sequence_length: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        if self._is_coordinate_array(structure_tokens):
            if isinstance(structure_tokens, torch.Tensor):
                return structure_tokens.detach().cpu().numpy()
            return np.asarray(structure_tokens)

        decoder_out = self.detokenize_tokens(structure_tokens, sequence_length)
        if decoder_out is None:
            return None

        coord_tensors = [
            decoder_out.get("atom37_positions"),
            decoder_out.get("final_atom_positions"),
            decoder_out.get("all_atom_positions"),
            decoder_out.get("positions"),
            decoder_out.get("coordinates"),
        ]
        coords = next((c for c in coord_tensors if c is not None), None)
        if coords is None:
            print(
                f"⚠️ No coordinate tensor found in detokenizer output keys: {list(decoder_out.keys())}"
            )
            return None

        if isinstance(coords, torch.Tensor):
            coords_np = coords.detach().cpu().numpy()
        else:
            coords_np = np.asarray(coords)

        if coords_np.ndim == 4:
            coords_np = coords_np[0]
        if coords_np.ndim == 3:
            atom_axis = 1 if coords_np.shape[1] >= 2 else 0
            return coords_np[:, 1, :] if atom_axis == 1 else coords_np[:, 0, :]
        if coords_np.ndim == 2:
            return coords_np

        print(f"⚠️ Unexpected coordinate shape: {coords_np.shape}")
        return None

    def tokens_to_pdb(
        self,
        structure_tokens: Union[str, List[int], np.ndarray, torch.Tensor],
        sequence: Optional[str],
        output_path: str,
        sequence_length: Optional[int] = None,
    ) -> bool:
        tokenizer = self.struct_tokenizer
        if tokenizer is None:
            return False

        decoder_out = self.detokenize_tokens(structure_tokens, sequence_length)
        if decoder_out is None:
            return False

        if sequence:
            from byprot.utils.protein.residue_constants import restypes

            aatype_ids = [
                restypes.index(aa) if aa in restypes else 20 for aa in sequence
            ]
            aatype_tensor = torch.tensor(
                [aatype_ids], dtype=torch.long, device=self.device
            )
            decoder_out["aatype"] = aatype_tensor

        header = Path(output_path).stem
        decoder_out["header"] = [header]

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            tokenizer.output_to_pdb(decoder_out, str(output_dir))
            print(f"✅ Saved structure to PDB: {output_path}")
            return True
        except Exception as exc:
            print(f"❌ PDB conversion failed: {exc}")
            return False

    def cleanup(self):
        if self._struct_tokenizer is not None:
            del self._struct_tokenizer
            self._struct_tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()



# Global converter instance
_converter = None

def get_structure_converter():
    """Get global structure converter instance"""
    global _converter
    if _converter is None:
        _converter = StructureTokenConverter()
    return _converter

def predict_structure_coords(model, tokenizer, sequence: str, return_plddt: bool = False):
    """
    Predict structure coordinates using ESMFold following evaluator_dplm2.py approach.
    
    Args:
        model: ESMFold model (esm.pretrained.esmfold_v1)
        tokenizer: Not used (ESM handles tokenization internally)  
        sequence: Protein sequence
        return_plddt: Whether to return pLDDT scores
        
    Returns:
        coordinates: (L, 3, 3) array for backbone atoms (N, CA, C)
        plddt_scores: (L,) array if return_plddt=True
    """
    try:
        if model is None:
            print("⚠️ ESMFold model is None")
            return None
        
        # Validate sequence length (ESMFold has limits)
        if len(sequence) > 400:
            print(f"⚠️ Sequence too long for ESMFold: {len(sequence)} residues")
            return None
        
        # Clean sequence following evaluator_dplm2.py approach
        # Replace unknown amino acids with alanine (line 73 in folding_model.py)
        cleaned_sequence = sequence.replace("X", "A")
        
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        final_sequence = ''.join([aa if aa in valid_aas else 'A' for aa in cleaned_sequence])
        
        if len(final_sequence) == 0:
            print(f"⚠️ Empty sequence after cleaning")
            return None
        
        print(f"   📝 ESMFold input: {final_sequence[:50]}..." if len(final_sequence) > 50 else f"   📝 ESMFold input: {final_sequence}")
        
        try:
            # Use ESM's official infer method (following folding_model.py line 75)
            torch.cuda.empty_cache()  # Clear cache before prediction
            
            with torch.no_grad():
                esmf_outputs = model.infer(final_sequence)
            
            # Extract coordinates from ESM output (different format than transformers)
            if "positions" in esmf_outputs:
                positions = esmf_outputs["positions"]
            elif "atom_positions" in esmf_outputs:
                positions = esmf_outputs["atom_positions"]
            else:
                print(f"⚠️ No position data in ESMFold output")
                print(f"   Available keys: {list(esmf_outputs.keys())}")
                return None
            
            print(f"   ✅ ESMFold output keys: {list(esmf_outputs.keys())}")
            print(f"   📊 Positions shape: {positions.shape}")
            
            # Handle ESM format: typically (batch, length, atoms, 3) or (length, atoms, 3)
            if len(positions.shape) == 4:  # (batch, length, atoms, 3)
                coords = positions[0]  # Take first batch
            elif len(positions.shape) == 3:  # (length, atoms, 3)
                coords = positions
            else:
                print(f"⚠️ Unexpected positions shape: {positions.shape}")
                return None
            
            # Extract backbone atoms (N, CA, C) - typically positions 0, 1, 2
            if coords.shape[1] >= 3:
                backbone_coords = coords[:, [0, 1, 2], :]  # (L, 3, 3)
                print(f"   ✅ Extracted backbone coordinates: {backbone_coords.shape}")
            else:
                print(f"⚠️ Not enough atoms in coords: {coords.shape}")
                return None
            
            # Convert to numpy
            backbone_coords_np = backbone_coords.cpu().numpy() if hasattr(backbone_coords, 'cpu') else backbone_coords
            
            if return_plddt:
                # Extract pLDDT scores (following folding_model.py line 79)
                plddt_scores = None
                if "mean_plddt" in esmf_outputs:
                    mean_plddt = esmf_outputs["mean_plddt"]
                    if hasattr(mean_plddt, 'item'):
                        mean_plddt = mean_plddt.item()
                    # Create per-residue pLDDT (simplified)
                    plddt_scores = np.full(len(final_sequence), mean_plddt / 100.0)  # Normalize to [0,1]
                elif "plddt" in esmf_outputs:
                    plddt_scores = esmf_outputs["plddt"]
                    if hasattr(plddt_scores, 'cpu'):
                        plddt_scores = plddt_scores.cpu().numpy()
                    if hasattr(plddt_scores, 'shape') and len(plddt_scores.shape) > 1:
                        plddt_scores = plddt_scores[0]  # Take first batch
                    # Normalize to [0,1] if needed
                    if plddt_scores is not None and plddt_scores.max() > 1.0:
                        plddt_scores = plddt_scores / 100.0
                
                print(f"   📊 pLDDT scores: {plddt_scores.shape if plddt_scores is not None else 'None'}")
                return backbone_coords_np, plddt_scores
            else:
                return backbone_coords_np
                
        except RuntimeError as e:
            if "CUDA error" in str(e) or "device-side assert" in str(e):
                print(f"⚠️ CUDA error in ESMFold - sequence may have issues")
                print(f"   Sequence length: {len(final_sequence)}")
                print(f"   Problematic chars: {set(sequence) - valid_aas}")
                return None
            else:
                raise
            
    except Exception as e:
        print(f"⚠️ Structure prediction failed: {e}")
        return None

def convert_structure_tokens_to_coords(
    structure_tokens: Union[str, List[int], np.ndarray, torch.Tensor],
    sequence_length: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Convenience function to convert structure tokens to coordinates"""
    converter = get_structure_converter()
    return converter.tokens_to_coordinates(structure_tokens, sequence_length)
