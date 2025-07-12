"""
plDDT Computation Module for Position-Level MCTS

This module provides plDDT (predicted Local Distance Difference Test) computation
using ESMFold for evaluating sequence confidence at each position.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PLDDTComputer:
    """
    Computes plDDT scores for sequences using ESMFold.
    
    plDDT is a measure of confidence in structure prediction at each position.
    Higher scores indicate more confident predictions.
    """
    
    def __init__(self, use_cpu: bool = False, chunk_size: Optional[int] = None):
        """
        Initialize the plDDT computer.
        
        Args:
            use_cpu: Whether to use CPU only (slower but more memory efficient)
            chunk_size: Chunk size for attention computation (None, 128, 64, 32)
        """
        self.use_cpu = use_cpu
        self.chunk_size = chunk_size
        self.model = None
        self.device = torch.device('cpu') if use_cpu else torch.device('cuda')
        
        # Cache for computed plDDT scores to avoid recomputation
        self.plddt_cache = {}
        
    def load_model(self):
        """Load the ESMFold model for structure prediction."""
        try:
            import esm
            logger.info("Loading ESMFold model...")
            
            self.model = esm.pretrained.esmfold_v1()
            self.model = self.model.eval()
            
            if self.chunk_size is not None:
                self.model.set_chunk_size(self.chunk_size)
            
            if self.use_cpu:
                self.model.esm.float()  # ESM-2 in fp16 not supported on CPU
                self.model.cpu()
            else:
                self.model.cuda()
                
            logger.info("ESMFold model loaded successfully")
            
        except ImportError:
            logger.warning("ESM not available, using fallback plDDT computation")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading ESMFold model: {e}")
            self.model = None
    
    def compute_plddt_scores(self, sequence: str, masked_positions: set) -> List[float]:
        """
        Compute plDDT scores for a sequence, focusing on masked positions.
        
        Args:
            sequence: Amino acid sequence (may contain 'X' for masked positions)
            masked_positions: Set of masked position indices
            
        Returns:
            List of plDDT scores for each position
        """
        if not masked_positions:
            return [1.0] * len(sequence)  # All positions unmasked
        
        # Check cache first
        cache_key = f"{sequence}_{tuple(sorted(masked_positions))}"
        if cache_key in self.plddt_cache:
            return self.plddt_cache[cache_key]
        
        try:
            if self.model is None:
                # Fallback to heuristic computation
                scores = self._compute_heuristic_plddt(sequence, masked_positions)
            else:
                # Use ESMFold for real plDDT computation
                scores = self._compute_esmfold_plddt(sequence, masked_positions)
            
            # Cache the result
            self.plddt_cache[cache_key] = scores
            return scores
            
        except Exception as e:
            logger.warning(f"Error computing plDDT scores: {e}")
            # Fallback to heuristic
            return self._compute_heuristic_plddt(sequence, masked_positions)
    
    def _compute_esmfold_plddt(self, sequence: str, masked_positions: set) -> List[float]:
        """
        Compute plDDT scores using ESMFold.
        
        Args:
            sequence: Amino acid sequence
            masked_positions: Set of masked position indices
            
        Returns:
            List of plDDT scores for each position
        """
        # Replace masked positions with random amino acids for structure prediction
        temp_sequence = list(sequence)
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        for pos in masked_positions:
            if pos < len(temp_sequence):
                temp_sequence[pos] = np.random.choice(list(amino_acids))
        
        temp_sequence = ''.join(temp_sequence)
        
        try:
            # Predict structure and get plDDT scores
            with torch.no_grad():
                output = self.model.infer([temp_sequence])
                
                # Extract per-residue plDDT scores
                plddt_scores = output["plddt"][0].cpu().numpy()  # Shape: (seq_len,)
                
                # Convert to list and ensure proper length
                scores = plddt_scores.tolist()
                
                # Pad or truncate to match sequence length
                if len(scores) < len(sequence):
                    scores.extend([0.0] * (len(sequence) - len(scores)))
                elif len(scores) > len(sequence):
                    scores = scores[:len(sequence)]
                
                return scores
                
        except Exception as e:
            logger.warning(f"ESMFold prediction failed: {e}")
            return self._compute_heuristic_plddt(sequence, masked_positions)
    
    def _compute_heuristic_plddt(self, sequence: str, masked_positions: set) -> List[float]:
        """
        Compute heuristic plDDT scores when ESMFold is not available.
        
        Args:
            sequence: Amino acid sequence
            masked_positions: Set of masked position indices
            
        Returns:
            List of heuristic plDDT scores for each position
        """
        scores = []
        
        for i in range(len(sequence)):
            if i in masked_positions:
                # Masked positions get low confidence
                base_score = 0.3
            else:
                # Unmasked positions get higher confidence
                base_score = 0.8
            
            # Add position-dependent variation
            # Positions near the middle tend to have higher confidence
            center_distance = abs(i - len(sequence) / 2) / len(sequence)
            position_factor = 1.0 - center_distance * 0.3
            
            # Add some randomness for exploration
            random_factor = np.random.uniform(-0.1, 0.1)
            
            # Combine factors
            score = base_score * position_factor + random_factor
            scores.append(max(0.0, min(1.0, score)))
        
        return scores
    
    def get_confidence_ranking(self, sequence: str, masked_positions: set) -> List[int]:
        """
        Get ranking of masked positions by confidence (plDDT score).
        
        Args:
            sequence: Amino acid sequence
            masked_positions: Set of masked position indices
            
        Returns:
            List of position indices ranked by confidence (highest first)
        """
        if not masked_positions:
            return []
        
        plddt_scores = self.compute_plddt_scores(sequence, masked_positions)
        
        # Create list of (position, score) tuples for masked positions
        position_scores = [(pos, plddt_scores[pos]) for pos in masked_positions if pos < len(plddt_scores)]
        
        # Sort by score (highest first)
        position_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the positions
        return [pos for pos, score in position_scores]
    
    def select_high_confidence_positions(
        self, 
        sequence: str, 
        masked_positions: set, 
        threshold: float = 0.7,
        max_positions: int = 3
    ) -> set:
        """
        Select masked positions with high confidence scores.
        
        Args:
            sequence: Amino acid sequence
            masked_positions: Set of masked position indices
            threshold: plDDT threshold for high confidence
            max_positions: Maximum number of positions to select
            
        Returns:
            Set of high-confidence masked positions
        """
        plddt_scores = self.compute_plddt_scores(sequence, masked_positions)
        
        high_confidence = set()
        for pos in masked_positions:
            if pos < len(plddt_scores) and plddt_scores[pos] > threshold:
                high_confidence.add(pos)
                if len(high_confidence) >= max_positions:
                    break
        
        return high_confidence
    
    def clear_cache(self):
        """Clear the plDDT score cache."""
        self.plddt_cache.clear()
        logger.info("plDDT cache cleared")


def create_plddt_computer(use_cpu: bool = False, chunk_size: Optional[int] = None) -> PLDDTComputer:
    """
    Create and initialize a plDDT computer.
    
    Args:
        use_cpu: Whether to use CPU only
        chunk_size: Chunk size for attention computation
        
    Returns:
        Initialized PLDDTComputer instance
    """
    computer = PLDDTComputer(use_cpu=use_cpu, chunk_size=chunk_size)
    computer.load_model()
    return computer


# Example usage and testing
def test_plddt_computation():
    """Test the plDDT computation functionality."""
    print("Testing plDDT computation...")
    
    # Create computer
    computer = create_plddt_computer(use_cpu=True)  # Use CPU for testing
    
    # Test sequence
    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    masked_positions = {10, 20, 30, 40, 50}
    
    print(f"Sequence: {sequence}")
    print(f"Masked positions: {masked_positions}")
    
    # Compute plDDT scores
    start_time = time.time()
    plddt_scores = computer.compute_plddt_scores(sequence, masked_positions)
    end_time = time.time()
    
    print(f"plDDT scores computed in {end_time - start_time:.2f}s")
    print(f"Average plDDT: {np.mean(plddt_scores):.3f}")
    print(f"Min plDDT: {np.min(plddt_scores):.3f}")
    print(f"Max plDDT: {np.max(plddt_scores):.3f}")
    
    # Test confidence ranking
    ranking = computer.get_confidence_ranking(sequence, masked_positions)
    print(f"Confidence ranking: {ranking}")
    
    # Test high confidence selection
    high_conf = computer.select_high_confidence_positions(sequence, masked_positions, threshold=0.6)
    print(f"High confidence positions: {high_conf}")
    
    return computer


if __name__ == "__main__":
    test_plddt_computation() 