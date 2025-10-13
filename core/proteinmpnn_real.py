#!/usr/bin/env python3
"""
Real ProteinMPNN Integration

This module integrates the actual ProteinMPNN model from denovo-protein-server
for inverse folding tasks in the MCTS pipeline.
"""

import os
import sys
import numpy as np
import torch
import tempfile
from typing import List, Optional, Dict, Any

# Add ProteinMPNN path
proteinmpnn_path = "/home/caom/AID3/dplm/denovo-protein-server/third_party/proteinpmnn"
sys.path.insert(0, proteinmpnn_path)

# Import real ProteinMPNN
from protein_mpnn_utils import ProteinMPNN, parse_PDB, StructureDatasetPDB, tied_featurize, _S_to_seq

class RealProteinMPNNExpert:
    """
    Real ProteinMPNN expert using the actual ProteinMPNN implementation.
    """
    
    def __init__(self, device: str = "cuda", temperature: float = 1.0, ca_only: bool = True):
        self.device = device
        self.temperature = temperature
        self.ca_only = ca_only
        self.expert_id = "proteinmpnn_real"
        self.model = None
        
        print(f"🧬 Real ProteinMPNN Expert initialized")
        print(f"   Device: {device}")
        print(f"   Temperature: {temperature}")
        print(f"   CA-only: {ca_only}")
    
    def load_model(self):
        """Load the real ProteinMPNN model."""
        try:
            print("🔄 Loading real ProteinMPNN model...")
            
            # Model configuration
            hidden_dim = 128
            num_layers = 3
            
            # Load model weights
            if self.ca_only:
                model_folder = os.path.join(proteinmpnn_path, "ca_model_weights")
                print("   Using CA-only ProteinMPNN model")
            else:
                model_folder = os.path.join(proteinmpnn_path, "vanilla_model_weights")
                print("   Using full-atom ProteinMPNN model")
            
            checkpoint_path = os.path.join(model_folder, "v_48_020.pt")
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"ProteinMPNN model weights not found: {checkpoint_path}")
            
            print(f"   Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Initialize model
            self.model = ProteinMPNN(
                ca_only=self.ca_only,
                num_letters=21,
                node_features=hidden_dim,
                edge_features=hidden_dim,
                hidden_dim=hidden_dim,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                augment_eps=0.0,
                k_neighbors=checkpoint['num_edges']
            )
            
            # Load state dict and move to device
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Real ProteinMPNN model loaded successfully")
            print(f"   Number of edges: {checkpoint['num_edges']}")
            print(f"   Training noise level: {checkpoint.get('noise_level', 'unknown')}A")
            
        except Exception as e:
            print(f"❌ Failed to load real ProteinMPNN model: {e}")
            raise
    
    def coords_to_pdb_string(self, coords: np.ndarray, sequence: str) -> str:
        """Convert coordinates to PDB string format."""
        pdb_lines = []
        pdb_lines.append("HEADER    PROTEIN                             01-JAN-00   TEST")
        pdb_lines.append("TITLE     TEST PROTEIN FOR REAL PROTEINMPNN")
        pdb_lines.append("MODEL        1")
        
        atom_id = 1
        for i, (coord, aa) in enumerate(zip(coords, sequence)):
            res_num = i + 1
            
            # Add CA atom - ensure proper formatting
            if not np.isnan(coord).any() and len(coord) == 3:
                pdb_lines.append(
                    f"ATOM  {atom_id:5d}  CA  {aa} A{res_num:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                    f"  1.00 20.00           C  "
                )
                atom_id += 1
        
        pdb_lines.append("ENDMDL")
        pdb_lines.append("END")
        return '\n'.join(pdb_lines)
    
    def generate_sequences_from_coords(self, masked_sequence: str, coords: np.ndarray, 
                                     num_samples: int = 1) -> List[str]:
        """
        Generate sequences directly from coordinates (like CAMEO .pkl data).
        
        Args:
            masked_sequence: Sequence with X for positions to design
            coords: CA coordinates array of shape (seq_len, 3) 
            num_samples: Number of sequences to generate
            
        Returns:
            List of generated sequences
        """
        try:
            if self.model is None:
                self.load_model()
            
            print(f"🎯 Real ProteinMPNN generating {num_samples} sequences from coordinates")
            print(f"   📊 Input coordinates shape: {coords.shape}")
            
            # Find masked positions
            mask_positions = [i for i, aa in enumerate(masked_sequence) if aa == 'X']
            print(f"   📍 Designing {len(mask_positions)} positions: {mask_positions[:10]}...")
            
            # Prepare data directly (bypass PDB parsing)
            seq_len = len(masked_sequence)
            
            # Create structure sequence (replace X with A for structure)
            structure_sequence = masked_sequence.replace('X', 'A')
            
            # Convert sequence to integer representation
            alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
            aa_to_int = {aa: i for i, aa in enumerate(alphabet)}
            S = torch.tensor([aa_to_int.get(aa, 20) for aa in structure_sequence], 
                           dtype=torch.long, device=self.device).unsqueeze(0)  # [1, seq_len]
            
            # Prepare coordinates tensor
            X = torch.tensor(coords, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, seq_len, 3]
            
            # Create mask (all positions are valid)
            mask = torch.ones(1, seq_len, dtype=torch.float32, device=self.device)
            
            # Create chain mask (single chain)
            chain_M = torch.ones(1, seq_len, dtype=torch.float32, device=self.device)
            
            # Create residue indices
            residue_idx = torch.arange(seq_len, device=self.device).unsqueeze(0)  # [1, seq_len]
            
            # Create chain encoding (all same chain)
            chain_encoding_all = torch.zeros(1, seq_len, dtype=torch.long, device=self.device)
            
            # Simple setup for other required parameters
            chain_M_pos = chain_M
            omit_AA_mask = torch.zeros(1, seq_len, 21, dtype=torch.float32, device=self.device)
            
            # Dummy values for unused parameters
            pssm_coef = torch.zeros(1, seq_len, dtype=torch.float32, device=self.device)
            pssm_bias = torch.zeros(1, seq_len, 21, dtype=torch.float32, device=self.device)
            bias_by_res_all = torch.zeros(1, seq_len, 21, dtype=torch.float32, device=self.device)
            
            print(f"   📊 Prepared tensors: X.shape={X.shape}, S.shape={S.shape}")
            
            sequences = []
            
            with torch.no_grad():
                # Generate sequences
                omit_AAs_np = np.zeros(21)  # Don't omit any amino acids  
                bias_AAs_np = np.zeros(21)  # No bias
                
                for i in range(num_samples):
                    randn = torch.randn(chain_M.shape, device=self.device)
                    
                    # Use ProteinMPNN sampling
                    sample_dict = self.model.sample(
                        X, randn, S, chain_M, chain_encoding_all, residue_idx,
                        mask=mask,
                        temperature=self.temperature,
                        omit_AAs_np=omit_AAs_np,
                        bias_AAs_np=bias_AAs_np,
                        chain_M_pos=chain_M_pos,
                        omit_AA_mask=omit_AA_mask,
                        pssm_coef=pssm_coef,
                        pssm_bias=pssm_bias,
                        pssm_multi=0.0,
                        pssm_log_odds_flag=False,
                        pssm_log_odds_mask=None,
                        pssm_bias_flag=False,
                        bias_by_res=bias_by_res_all
                    )
                    
                    S_sample = sample_dict["S"]
                    
                    # Convert to sequence string
                    seq = _S_to_seq(S_sample[0], chain_M[0])
                    sequences.append(seq)
                    
                    print(f"   Generated seq {i+1}: {seq[:50]}... (length: {len(seq)})")
            
            print(f"✅ Real ProteinMPNN generated {len(sequences)} sequences")
            return sequences
                
        except Exception as e:
            print(f"❌ Real ProteinMPNN generation failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def generate_sequences(self, masked_sequence: str, num_samples: int = 1, 
                          structure_coords: Optional[np.ndarray] = None) -> List[str]:
        """
        Generate sequences using real ProteinMPNN (wrapper for backward compatibility).
        """
        if structure_coords is not None:
            return self.generate_sequences_from_coords(masked_sequence, structure_coords, num_samples)
        else:
            print("❌ Real ProteinMPNN requires structure coordinates")
            return []
    
    def generate(self, masked_sequence: str = None, num_samples: int = 1, 
                structure_coords: Optional[np.ndarray] = None, **kwargs) -> List[str]:
        """
        Generate method for compatibility with MCTS pipeline.
        Alias for generate_sequences.
        """
        return self.generate_sequences(masked_sequence, num_samples, structure_coords)
    
    def compute_entropy(self, masked_sequence: str, structure_coords: Optional[np.ndarray] = None) -> float:
        """
        Compute predictive entropy using real ProteinMPNN logits.
        
        Returns higher entropy for more uncertain/difficult positions.
        """
        try:
            if self.model is None:
                self.load_model()
            
            # Count masked positions
            mask_count = masked_sequence.count('X')
            
            if mask_count == 0:
                return 0.1  # Low entropy for no masking
            
            if structure_coords is None:
                # No structure info - entropy based on mask count only
                entropy = min(3.0, 0.5 + 0.15 * mask_count)
                return entropy
            
            print(f"🔄 Computing real ProteinMPNN entropy from model logits...")
            
            # Prepare data for ProteinMPNN forward pass
            seq_len = len(masked_sequence)
            structure_sequence = masked_sequence.replace('X', 'A')  # Placeholder for structure
            
            # Convert sequence to integer representation
            alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
            aa_to_int = {aa: i for i, aa in enumerate(alphabet)}
            S = torch.tensor([aa_to_int.get(aa, 20) for aa in structure_sequence], 
                           dtype=torch.long, device=self.device).unsqueeze(0)  # [1, seq_len]
            
            # Prepare coordinates tensor
            X = torch.tensor(structure_coords, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, seq_len, 3]
            
            # Create mask and other required tensors
            mask = torch.ones(1, seq_len, dtype=torch.float32, device=self.device)
            chain_M = torch.ones(1, seq_len, dtype=torch.float32, device=self.device)
            residue_idx = torch.arange(seq_len, device=self.device).unsqueeze(0)
            chain_encoding_all = torch.zeros(1, seq_len, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                # Create randn tensor (required by ProteinMPNN forward)
                randn = torch.randn(chain_M.shape, device=self.device)
                
                # Get logits from ProteinMPNN model
                log_probs = self.model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
                
                # Convert log probabilities to probabilities
                probs = torch.exp(log_probs)  # [1, seq_len, 21]
                
                # Calculate entropy for masked positions only
                mask_positions = [i for i, aa in enumerate(masked_sequence) if aa == 'X']
                
                total_entropy = 0.0
                for pos in mask_positions:
                    if pos < probs.shape[1]:
                        # Get probability distribution for this position
                        pos_probs = probs[0, pos, :]  # [21]
                        
                        # Calculate entropy: H = -sum(p * log(p))
                        # Add small epsilon to avoid log(0)
                        epsilon = 1e-8
                        pos_probs = pos_probs + epsilon
                        pos_entropy = -torch.sum(pos_probs * torch.log(pos_probs))
                        total_entropy += pos_entropy.item()
                
                # Average entropy across masked positions
                avg_entropy = total_entropy / len(mask_positions) if mask_positions else 0.0
                
                print(f"   📊 Real ProteinMPNN entropy: {avg_entropy:.3f} (averaged over {len(mask_positions)} positions)")
                
                return avg_entropy
            
        except Exception as e:
            print(f"⚠️ Real ProteinMPNN entropy calculation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to heuristic
            mask_count = masked_sequence.count('X')
            return min(3.0, 0.5 + 0.1 * mask_count)
    
    def get_expert_id(self) -> str:
        """Return expert identifier."""
        return self.expert_id
    
    def unload_model(self):
        """Clean up model."""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🗑️ Real ProteinMPNN Expert unloaded")


def create_real_proteinmpnn_expert(device: str = "cuda", temperature: float = 1.0, ca_only: bool = True) -> RealProteinMPNNExpert:
    """Create a real ProteinMPNN expert instance."""
    return RealProteinMPNNExpert(device=device, temperature=temperature, ca_only=ca_only)


# Test the real expert
if __name__ == "__main__":
    print("🧪 Testing Real ProteinMPNN Expert")
    
    # Create expert
    expert = create_real_proteinmpnn_expert()
    expert.load_model()
    
    # Test with dummy coordinates
    test_coords = np.random.randn(20, 3) * 10
    test_sequence = "MKLLVLGLGAGVGKSALTIQ"
    
    # Test generation
    masked_seq = "MKLLVLXXXAGVGKSALTIQ"
    generated = expert.generate_sequences(masked_seq, num_samples=2, structure_coords=test_coords)
    
    print(f"📝 Original: {test_sequence}")
    print(f"🎭 Masked:   {masked_seq}")
    for i, seq in enumerate(generated):
        print(f"✨ Gen {i+1}:   {seq}")
    
    # Test entropy
    entropy = expert.compute_entropy(masked_seq, test_coords)
    print(f"📊 Entropy: {entropy:.3f}")
    
    expert.unload_model()
    print("✅ Real ProteinMPNN Expert test completed")
