#!/usr/bin/env python3

"""
Structure token to coordinate conversion utility
Based on the official DPLM-2 evaluator pipeline
"""

import torch
import numpy as np
from typing import Optional, Union, List
import tempfile
import os
from pathlib import Path

class StructureTokenConverter:
    """Convert DPLM-2 structure tokens to coordinates using the official pipeline"""
    
    def __init__(self):
        self._struct_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def struct_tokenizer(self):
        """Load structure tokenizer lazily"""
        if self._struct_tokenizer is None:
            try:
                from byprot.models.utils import get_struct_tokenizer
                print(f"🔧 Loading structure tokenizer...")
                self._struct_tokenizer = get_struct_tokenizer().to(self.device)
                print(f"✅ Structure tokenizer loaded")
            except Exception as e:
                print(f"❌ Failed to load structure tokenizer: {e}")
                return None
        return self._struct_tokenizer
    
    def tokens_to_coordinates(self, structure_tokens: Union[str, List[int]], sequence_length: int) -> Optional[np.ndarray]:
        """
        Convert structure tokens to 3D coordinates
        
        Args:
            structure_tokens: Structure tokens as string or list of integers
            sequence_length: Length of the corresponding amino acid sequence
            
        Returns:
            numpy array of shape (N, 3) with CA coordinates, or None if failed
        """
        try:
            if self.struct_tokenizer is None:
                return None
            
            # Parse structure tokens
            if isinstance(structure_tokens, str):
                if "," in structure_tokens:
                    token_ids = [int(t.strip()) for t in structure_tokens.split(",") if t.strip().isdigit()]
                else:
                    # Handle space-separated or continuous tokens
                    token_ids = [int(t) for t in structure_tokens.split() if t.isdigit()]
                    if not token_ids:
                        # Try parsing as continuous string of digits
                        token_ids = [int(structure_tokens[i:i+3]) for i in range(0, len(structure_tokens), 3) 
                                   if structure_tokens[i:i+3].isdigit()]
            else:
                token_ids = structure_tokens
            
            if not token_ids:
                print(f"⚠️ No valid tokens found in: {structure_tokens}")
                return None
            
            # Ensure we have the right number of tokens for the sequence length
            expected_tokens = sequence_length + 2  # +2 for CLS and EOS tokens
            if len(token_ids) != expected_tokens:
                print(f"⚠️ Token count mismatch: got {len(token_ids)}, expected {expected_tokens}")
                # Truncate or pad as needed
                if len(token_ids) > expected_tokens:
                    token_ids = token_ids[:expected_tokens]
                else:
                    # Pad with a common structure token (e.g., 159)
                    token_ids.extend([159] * (expected_tokens - len(token_ids)))
            
            # Convert to tensor format expected by detokenize
            structok = torch.tensor([token_ids], dtype=torch.long).to(self.device)  # Add batch dimension
            res_mask = torch.ones(structok.shape, dtype=torch.float).to(self.device)
            
            # Use detokenize method (same as official evaluation)
            with torch.no_grad():
                decoder_out = self.struct_tokenizer.detokenize(structok, res_mask)
            
            # Extract coordinates from decoder output
            if isinstance(decoder_out, dict):
                # Look for coordinate keys in order of preference
                coord_keys = ['all_atom_positions', 'positions', 'coordinates', 'atom_positions']
                coords = None
                
                for key in coord_keys:
                    if key in decoder_out:
                        coords = decoder_out[key]
                        break
                
                if coords is not None:
                    # Convert to numpy and extract CA coordinates
                    if isinstance(coords, torch.Tensor):
                        coords = coords.detach().cpu().numpy()
                    
                    # Handle different coordinate formats
                    if coords.ndim == 4:  # (batch, residues, atoms, 3)
                        coords = coords[0, :, 1, :]  # CA atom (index 1)
                    elif coords.ndim == 3:  # (residues, atoms, 3)
                        coords = coords[:, 1, :] if coords.shape[1] > 1 else coords[:, 0, :]
                    elif coords.ndim == 2:  # Already (residues, 3)
                        pass
                    
                    print(f"🔧 Converted structure tokens to coordinates: {coords.shape}")
                    return coords
                else:
                    print(f"🔧 Available decoder output keys: {list(decoder_out.keys())}")
                    return None
            else:
                print(f"⚠️ Unexpected decoder output type: {type(decoder_out)}")
                return None
                
        except Exception as e:
            print(f"❌ Structure token conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def tokens_to_pdb(self, structure_tokens: Union[str, List[int]], sequence: str, output_path: str) -> bool:
        """
        Convert structure tokens to PDB file using official pipeline
        
        Args:
            structure_tokens: Structure tokens as string or list of integers
            sequence: Amino acid sequence
            output_path: Path to save PDB file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.struct_tokenizer is None:
                return False
            
            # Parse structure tokens
            if isinstance(structure_tokens, str):
                if "," in structure_tokens:
                    token_ids = [int(t.strip()) for t in structure_tokens.split(",") if t.strip().isdigit()]
                else:
                    token_ids = [int(t) for t in structure_tokens.split() if t.isdigit()]
            else:
                token_ids = structure_tokens
            
            # Convert sequence to aatype indices
            from byprot.utils.protein.residue_constants import restypes
            aatype = []
            for aa in sequence:
                if aa in restypes:
                    aatype.append(restypes.index(aa))
                else:
                    aatype.append(20)  # Unknown amino acid
            
            # Prepare batch data
            structok = torch.tensor([token_ids], dtype=torch.long).to(self.device)
            res_mask = torch.ones(structok.shape, dtype=torch.float).to(self.device)
            aatype_tensor = torch.tensor([aatype], dtype=torch.long).to(self.device)
            
            # Create batch dictionary
            batch = {
                'structok': structok,
                'res_mask': res_mask,
                'aatype': aatype_tensor,
                'header': [Path(output_path).stem]
            }
            
            # Use detokenize method
            with torch.no_grad():
                decoder_out = self.struct_tokenizer.detokenize(batch["structok"], batch["res_mask"])
            
            # Add required fields
            decoder_out["aatype"] = batch["aatype"]
            decoder_out["header"] = batch["header"]
            
            # Save to PDB using official method
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.struct_tokenizer.output_to_pdb(decoder_out, str(output_dir))
            
            print(f"✅ Saved structure to PDB: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ PDB conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """Clean up resources"""
        if self._struct_tokenizer is not None:
            del self._struct_tokenizer
            self._struct_tokenizer = None
            torch.cuda.empty_cache()

# Global converter instance
_converter = None

def get_structure_converter():
    """Get global structure converter instance"""
    global _converter
    if _converter is None:
        _converter = StructureTokenConverter()
    return _converter

def convert_structure_tokens_to_coords(structure_tokens: Union[str, List[int]], sequence_length: int) -> Optional[np.ndarray]:
    """Convenience function to convert structure tokens to coordinates"""
    converter = get_structure_converter()
    return converter.tokens_to_coordinates(structure_tokens, sequence_length)
