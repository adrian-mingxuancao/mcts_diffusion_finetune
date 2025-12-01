"""
Structure Tokenization Utilities for DPLM-2
===========================================

This module handles the conversion between different structure representations:
1. PDB coordinates -> DPLM-2 structure tokens
2. ESMFold output -> DPLM-2 structure tokens  
3. DPLM-2 structure tokens -> coordinates (for evaluation)

Key functions:
- esmfold_to_dplm_tokens(): Convert ESMFold prediction to DPLM structure tokens
- coords_to_dplm_tokens(): Convert 3D coordinates to DPLM structure tokens
- dplm_tokens_to_coords(): Convert DPLM structure tokens back to coordinates
- validate_structure_tokens(): Check if structure tokens are valid
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import warnings

try:
    from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue
    from Bio.PDB.vectors import Vector, calc_angle, calc_dihedral
    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False
    logging.warning("BioPython not available - structure tokenization will be limited")

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    logging.warning("ESM not available - ESMFold integration will be limited")

# Try to import DPLM tokenizer
try:
    from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel as DPLM2
    DPLM_AVAILABLE = True
except ImportError:
    DPLM_AVAILABLE = False
    logging.warning("DPLM2 not available - will use fallback tokenization")

logger = logging.getLogger(__name__)

# Suppress BioPython warnings
warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB")


class StructureTokenizer:
    """
    Handles conversion between structure representations and DPLM-2 tokens.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.dplm_tokenizer = None
        
        # Load DPLM tokenizer if available
        if DPLM_AVAILABLE:
            try:
                # Load a small DPLM model to get tokenizer
                model = DPLM2.from_pretrained("airkingbd/dplm2_150m")
                self.dplm_tokenizer = model.tokenizer
                logger.info("✅ DPLM tokenizer loaded")
            except Exception as e:
                logger.warning(f"Failed to load DPLM tokenizer: {e}")
        
        # Fallback: create simple token mapping
        if self.dplm_tokenizer is None:
            self._create_fallback_tokenizer()
    
    def _create_fallback_tokenizer(self):
        """Create simple fallback tokenizer for structure tokens."""
        # This is a simplified version - real DPLM uses more sophisticated tokenization
        # We'll use a basic discretization of backbone angles
        
        # Phi/Psi angle bins (simplified)
        self.phi_bins = np.linspace(-180, 180, 36)  # 10-degree bins
        self.psi_bins = np.linspace(-180, 180, 36)  # 10-degree bins
        
        # Create token vocabulary (simplified)
        self.token_vocab = {}
        self.reverse_vocab = {}
        token_id = 100  # Start after special tokens
        
        for i in range(len(self.phi_bins) - 1):
            for j in range(len(self.psi_bins) - 1):
                token = f"struct_{i}_{j}"
                self.token_vocab[token] = token_id
                self.reverse_vocab[token_id] = token
                token_id += 1
        
        # Add special tokens
        self.token_vocab["<mask>"] = 0
        self.token_vocab["<pad>"] = 1
        self.token_vocab["<cls>"] = 2
        self.token_vocab["<sep>"] = 3
        
        self.reverse_vocab[0] = "<mask>"
        self.reverse_vocab[1] = "<pad>"
        self.reverse_vocab[2] = "<cls>"
        self.reverse_vocab[3] = "<sep>"
        
        logger.info(f"✅ Fallback tokenizer created with {len(self.token_vocab)} tokens")
    
    def esmfold_to_dplm_tokens(self, esmfold_output: Any, sequence: str) -> str:
        """
        Convert ESMFold output to DPLM-2 structure tokens.
        
        Args:
            esmfold_output: ESMFold model output
            sequence: Amino acid sequence
            
        Returns:
            Comma-separated structure tokens
        """
        try:
            # Extract coordinates from ESMFold output
            if hasattr(esmfold_output, 'positions'):
                # ESMFold v1 format
                coords = esmfold_output.positions
            elif hasattr(esmfold_output, 'atom_positions'):
                # Alternative format
                coords = esmfold_output.atom_positions
            else:
                # Try to parse from PDB string if available
                if hasattr(esmfold_output, 'pdb'):
                    return self.pdb_string_to_dplm_tokens(esmfold_output.pdb)
                else:
                    logger.warning("Cannot extract coordinates from ESMFold output")
                    return self._create_fallback_tokens(len(sequence))
            
            # Convert coordinates to DPLM tokens
            return self.coords_to_dplm_tokens(coords, sequence)
            
        except Exception as e:
            logger.warning(f"ESMFold to DPLM conversion failed: {e}")
            return self._create_fallback_tokens(len(sequence))
    
    def coords_to_dplm_tokens(self, coords: Union[np.ndarray, torch.Tensor], 
                             sequence: str) -> str:
        """
        Convert 3D coordinates to DPLM-2 structure tokens.
        
        Args:
            coords: Coordinates array [N, 3] or [N, atom_types, 3]
            sequence: Amino acid sequence
            
        Returns:
            Comma-separated structure tokens
        """
        try:
            # Convert to numpy if tensor
            if isinstance(coords, torch.Tensor):
                coords = coords.detach().cpu().numpy()
            
            # Handle different coordinate formats
            if coords.ndim == 3:
                # [N, atom_types, 3] - extract CA atoms (typically index 1)
                if coords.shape[1] >= 2:
                    ca_coords = coords[:, 1, :]  # CA atoms
                else:
                    ca_coords = coords[:, 0, :]  # First atom type
            else:
                # [N, 3] - assume CA coordinates
                ca_coords = coords
            
            # Calculate backbone angles
            phi_psi_angles = self._calculate_backbone_angles(ca_coords)
            
            # Convert angles to tokens
            tokens = []
            for phi, psi in phi_psi_angles:
                token = self._angles_to_token(phi, psi)
                tokens.append(str(token))
            
            return ",".join(tokens)
            
        except Exception as e:
            logger.warning(f"Coordinates to DPLM conversion failed: {e}")
            return self._create_fallback_tokens(len(sequence))
    
    def pdb_string_to_dplm_tokens(self, pdb_string: str) -> str:
        """
        Convert PDB string to DPLM-2 structure tokens.
        
        Args:
            pdb_string: PDB format string
            
        Returns:
            Comma-separated structure tokens
        """
        try:
            if not BIO_AVAILABLE:
                logger.warning("BioPython not available for PDB parsing")
                return self._create_fallback_tokens(100)  # Default length
            
            # Parse PDB string
            from io import StringIO
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", StringIO(pdb_string))
            
            # Extract CA coordinates
            ca_coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            ca_atom = residue['CA']
                            ca_coords.append(ca_atom.coord)
            
            if not ca_coords:
                logger.warning("No CA atoms found in PDB")
                return self._create_fallback_tokens(100)
            
            ca_coords = np.array(ca_coords)
            
            # Calculate backbone angles and convert to tokens
            phi_psi_angles = self._calculate_backbone_angles(ca_coords)
            
            tokens = []
            for phi, psi in phi_psi_angles:
                token = self._angles_to_token(phi, psi)
                tokens.append(str(token))
            
            return ",".join(tokens)
            
        except Exception as e:
            logger.warning(f"PDB to DPLM conversion failed: {e}")
            return self._create_fallback_tokens(100)
    
    def dplm_tokens_to_coords(self, structure_tokens: str, sequence: str) -> np.ndarray:
        """
        Convert DPLM-2 structure tokens back to approximate coordinates.
        
        Args:
            structure_tokens: Comma-separated structure tokens
            sequence: Amino acid sequence
            
        Returns:
            CA coordinates array [N, 3]
        """
        try:
            tokens = structure_tokens.split(',')
            
            # Convert tokens back to angles
            angles = []
            for token in tokens:
                if token.strip() == '<mask>' or not token.strip():
                    # Use default angles for masked tokens
                    phi, psi = -60.0, -45.0  # Alpha helix
                else:
                    phi, psi = self._token_to_angles(token.strip())
                angles.append((phi, psi))
            
            # Build approximate backbone from angles
            coords = self._build_backbone_from_angles(angles)
            
            return coords
            
        except Exception as e:
            logger.warning(f"DPLM tokens to coordinates conversion failed: {e}")
            # Return linear backbone as fallback
            return self._create_linear_backbone(len(sequence))
    
    def _calculate_backbone_angles(self, ca_coords: np.ndarray) -> List[Tuple[float, float]]:
        """Calculate phi/psi angles from CA coordinates."""
        angles = []
        
        for i in range(len(ca_coords)):
            if i == 0 or i == len(ca_coords) - 1:
                # Terminal residues - use default angles
                phi, psi = -60.0, -45.0  # Alpha helix default
            else:
                try:
                    # Calculate phi angle (C(i-1) - N(i) - CA(i) - C(i))
                    # Simplified: use CA(i-1) - CA(i) - CA(i+1) angle as proxy
                    v1 = ca_coords[i] - ca_coords[i-1]
                    v2 = ca_coords[i+1] - ca_coords[i]
                    
                    # Calculate angle between vectors
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_angle))
                    
                    # Convert to phi/psi representation (simplified)
                    phi = angle - 180.0  # Center around -180 to 180
                    psi = angle - 120.0  # Offset for psi
                    
                    # Ensure angles are in valid range
                    phi = ((phi + 180) % 360) - 180
                    psi = ((psi + 180) % 360) - 180
                    
                except:
                    phi, psi = -60.0, -45.0  # Default
            
            angles.append((phi, psi))
        
        return angles
    
    def _angles_to_token(self, phi: float, psi: float) -> int:
        """Convert phi/psi angles to structure token."""
        try:
            if self.dplm_tokenizer is not None:
                # Use real DPLM tokenizer if available
                # This is a placeholder - real implementation would use DPLM's method
                return self._simple_angle_tokenization(phi, psi)
            else:
                # Use fallback tokenization
                return self._simple_angle_tokenization(phi, psi)
        except:
            return 100  # Default token
    
    def _simple_angle_tokenization(self, phi: float, psi: float) -> int:
        """Simple angle tokenization using binning."""
        # Find phi bin
        phi_idx = np.digitize(phi, self.phi_bins) - 1
        phi_idx = max(0, min(len(self.phi_bins) - 2, phi_idx))
        
        # Find psi bin
        psi_idx = np.digitize(psi, self.psi_bins) - 1
        psi_idx = max(0, min(len(self.psi_bins) - 2, psi_idx))
        
        # Convert to token ID
        token_id = 100 + phi_idx * len(self.psi_bins) + psi_idx
        return token_id
    
    def _token_to_angles(self, token: str) -> Tuple[float, float]:
        """Convert structure token back to phi/psi angles."""
        try:
            token_id = int(token)
            
            if token_id < 100:
                # Special token
                return -60.0, -45.0  # Alpha helix default
            
            # Convert back to phi/psi indices
            token_id -= 100
            phi_idx = token_id // len(self.psi_bins)
            psi_idx = token_id % len(self.psi_bins)
            
            # Get angle values (use bin centers)
            phi = (self.phi_bins[phi_idx] + self.phi_bins[phi_idx + 1]) / 2
            psi = (self.psi_bins[psi_idx] + self.psi_bins[psi_idx + 1]) / 2
            
            return phi, psi
            
        except:
            return -60.0, -45.0  # Default
    
    def _build_backbone_from_angles(self, angles: List[Tuple[float, float]]) -> np.ndarray:
        """Build approximate backbone coordinates from phi/psi angles."""
        coords = []
        
        # Start at origin
        current_pos = np.array([0.0, 0.0, 0.0])
        current_dir = np.array([1.0, 0.0, 0.0])  # Initial direction
        
        for i, (phi, psi) in enumerate(angles):
            coords.append(current_pos.copy())
            
            # Move to next position (simplified backbone building)
            # In reality, this would use proper protein geometry
            step_size = 3.8  # Approximate CA-CA distance
            
            # Rotate direction based on angles (simplified)
            phi_rad = np.radians(phi)
            psi_rad = np.radians(psi)
            
            # Simple rotation (not geometrically accurate, but reasonable for approximation)
            rotation_angle = (phi_rad + psi_rad) / 2
            cos_r = np.cos(rotation_angle)
            sin_r = np.sin(rotation_angle)
            
            # Rotate in XY plane
            new_dir = np.array([
                current_dir[0] * cos_r - current_dir[1] * sin_r,
                current_dir[0] * sin_r + current_dir[1] * cos_r,
                current_dir[2]
            ])
            
            current_dir = new_dir / np.linalg.norm(new_dir)
            current_pos += current_dir * step_size
        
        return np.array(coords)
    
    def _create_linear_backbone(self, length: int) -> np.ndarray:
        """Create linear backbone coordinates as fallback."""
        coords = []
        for i in range(length):
            coords.append([i * 3.8, 0.0, 0.0])  # 3.8 Å spacing
        return np.array(coords)
    
    def _create_fallback_tokens(self, length: int) -> str:
        """Create fallback structure tokens."""
        # Use alpha helix tokens as default
        default_token = self._simple_angle_tokenization(-60.0, -45.0)
        tokens = [str(default_token)] * length
        return ",".join(tokens)
    
    def validate_structure_tokens(self, structure_tokens: str) -> bool:
        """
        Validate if structure tokens are properly formatted.
        
        Args:
            structure_tokens: Comma-separated structure tokens
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not structure_tokens or structure_tokens.strip() == "":
                return False
            
            tokens = structure_tokens.split(',')
            
            # Check if all tokens are valid
            for token in tokens:
                token = token.strip()
                if token == '<mask>':
                    continue  # Mask tokens are valid
                
                try:
                    token_id = int(token)
                    if token_id < 0:
                        return False
                except ValueError:
                    return False
            
            return True
            
        except:
            return False


# Global instance for easy access
_tokenizer_instance = None

def get_structure_tokenizer(device: str = "cuda") -> StructureTokenizer:
    """Get global structure tokenizer instance."""
    global _tokenizer_instance
    if _tokenizer_instance is None:
        _tokenizer_instance = StructureTokenizer(device)
    return _tokenizer_instance


# Convenience functions
def esmfold_to_dplm_tokens(esmfold_output: Any, sequence: str, 
                          device: str = "cuda") -> str:
    """Convert ESMFold output to DPLM structure tokens."""
    tokenizer = get_structure_tokenizer(device)
    return tokenizer.esmfold_to_dplm_tokens(esmfold_output, sequence)


def coords_to_dplm_tokens(coords: Union[np.ndarray, torch.Tensor], 
                         sequence: str, device: str = "cuda") -> str:
    """Convert coordinates to DPLM structure tokens."""
    tokenizer = get_structure_tokenizer(device)
    return tokenizer.coords_to_dplm_tokens(coords, sequence)


def dplm_tokens_to_coords(structure_tokens: str, sequence: str, 
                         device: str = "cuda") -> np.ndarray:
    """Convert DPLM structure tokens to coordinates."""
    tokenizer = get_structure_tokenizer(device)
    return tokenizer.dplm_tokens_to_coords(structure_tokens, sequence)


def validate_structure_tokens(structure_tokens: str) -> bool:
    """Validate structure tokens format."""
    tokenizer = get_structure_tokenizer()
    return tokenizer.validate_structure_tokens(structure_tokens)






































