"""
Task-Specific Evaluators for Unified MCTS Framework
==================================================

This module provides evaluators for different protein modeling tasks:
- FoldingEvaluator: Evaluates structure prediction quality
- InverseFoldingEvaluator: Evaluates sequence design quality  
- MotifScaffoldingEvaluator: Evaluates motif completion quality

Each evaluator implements task-specific metrics and reward computation.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
from abc import ABC, abstractmethod

try:
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Structure import Structure
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False
    logging.warning("BioPython not available - some evaluations will be limited")

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    logging.warning("ESM not available - ESMFold evaluations will be limited")

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Base class for task-specific evaluators."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.cache = {}
    
    def evaluate(self, **kwargs) -> float:
        """Evaluate and return reward score."""
        # Default implementation - subclasses should override
        return 0.5


class FoldingEvaluator(BaseEvaluator):
    """
    Evaluator for protein folding task.
    
    Given amino acid sequence and predicted structure tokens,
    evaluates structural quality using multiple metrics:
    - Structure validity (parseable, reasonable geometry)
    - Biophysical properties (hydrophobicity, charge distribution)
    - Confidence scores (pLDDT if available)
    """
    
    def __init__(self, reference_structure: Optional[np.ndarray] = None, 
                 device: str = "cuda"):
        """
        Initialize folding evaluator.
        
        Args:
            reference_structure: Reference structure coordinates for comparison (optional)
            device: CUDA device
        """
        super().__init__(device)
        self.reference_structure = reference_structure
        
        # Load ESMFold for structure validation if available
        self.esmfold_model = None
        if ESM_AVAILABLE:
            try:
                self.esmfold_model = esm.pretrained.esmfold_v1()
                self.esmfold_model = self.esmfold_model.eval().to(device)
                logger.info("✅ ESMFold loaded for structure validation")
            except Exception as e:
                logger.warning(f"Failed to load ESMFold: {e}")
    
    def evaluate_structure(self, sequence: str, structure_tokens: str) -> float:
        """
        Evaluate structure prediction quality.
        
        Args:
            sequence: Amino acid sequence
            structure_tokens: Predicted structure tokens
            
        Returns:
            Reward score (higher is better)
        """
        if not sequence or not structure_tokens:
            return 0.0
        
        # Create cache key
        cache_key = f"fold_{hash(sequence)}_{hash(structure_tokens)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        reward = 0.0
        
        try:
            # 1. Structure token validity
            validity_score = self._evaluate_structure_validity(structure_tokens)
            reward += validity_score * 0.3
            
            # 2. Sequence-structure compatibility
            compatibility_score = self._evaluate_sequence_structure_compatibility(
                sequence, structure_tokens
            )
            reward += compatibility_score * 0.4
            
            # 3. Biophysical properties
            biophysical_score = self._evaluate_biophysical_properties(sequence)
            reward += biophysical_score * 0.2
            
            # 4. Reference structure comparison (if available)
            if self.reference_structure is not None:
                structural_similarity = self._evaluate_structural_similarity(
                    sequence, structure_tokens
                )
                reward += structural_similarity * 0.1
            
            # Normalize reward to [0, 1]
            reward = max(0.0, min(1.0, reward))
            
        except Exception as e:
            logger.warning(f"Structure evaluation failed: {e}")
            reward = 0.0
        
        # Cache result
        self.cache[cache_key] = reward
        return reward
    
    def _evaluate_structure_validity(self, structure_tokens: str) -> float:
        """Evaluate if structure tokens are valid and reasonable."""
        try:
            # Parse structure tokens
            tokens = structure_tokens.split(',')
            
            # Check for reasonable token count
            if len(tokens) < 5:
                return 0.1  # Too short
            
            # Check for mask tokens (should be resolved)
            mask_ratio = sum(1 for token in tokens if '<mask>' in token) / len(tokens)
            if mask_ratio > 0.5:
                return 0.2  # Too many unresolved masks
            
            # Check for token diversity (not all the same)
            unique_tokens = len(set(tokens))
            diversity_score = min(1.0, unique_tokens / (len(tokens) * 0.3))
            
            return 0.5 + 0.5 * diversity_score
            
        except Exception:
            return 0.1
    
    def _evaluate_sequence_structure_compatibility(self, sequence: str, 
                                                 structure_tokens: str) -> float:
        """Evaluate if sequence and structure are compatible."""
        try:
            # Check length compatibility
            seq_length = len(sequence)
            struct_length = len(structure_tokens.split(','))
            
            length_ratio = min(seq_length, struct_length) / max(seq_length, struct_length)
            if length_ratio < 0.8:
                return 0.2  # Length mismatch
            
            # Use ESMFold for validation if available
            if self.esmfold_model is not None:
                return self._evaluate_with_esmfold(sequence)
            
            # Fallback: simple heuristic based on sequence properties
            return self._evaluate_sequence_structure_heuristic(sequence, structure_tokens)
            
        except Exception:
            return 0.3
    
    def _evaluate_with_esmfold(self, sequence: str) -> float:
        """Use ESMFold to evaluate sequence foldability."""
        try:
            with torch.no_grad():
                # Predict structure using ESMFold
                output = self.esmfold_model.infer_pdb(sequence)
                
                # Extract confidence scores (pLDDT)
                if hasattr(output, 'plddt'):
                    plddt_scores = output.plddt
                    mean_plddt = float(plddt_scores.mean())
                    
                    # Convert pLDDT to reward (0-100 -> 0-1)
                    return mean_plddt / 100.0
                
                # Fallback: assume reasonable quality if no error
                return 0.7
                
        except Exception as e:
            logger.debug(f"ESMFold evaluation failed: {e}")
            return 0.5
    
    def _evaluate_sequence_structure_heuristic(self, sequence: str, 
                                             structure_tokens: str) -> float:
        """Simple heuristic for sequence-structure compatibility."""
        # Check for reasonable amino acid composition
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        # Penalize if dominated by single amino acid
        max_freq = max(aa_counts.values()) / len(sequence)
        if max_freq > 0.5:
            return 0.3
        
        # Reward diverse composition
        diversity = len(aa_counts) / 20  # 20 standard amino acids
        return 0.5 + 0.3 * diversity
    
    def _evaluate_biophysical_properties(self, sequence: str) -> float:
        """Evaluate biophysical properties of the sequence."""
        try:
            if not BIO_AVAILABLE:
                return 0.5  # Neutral score if BioPython unavailable
            
            # Use BioPython's ProteinAnalysis
            analysis = ProteinAnalysis(sequence)
            
            # Check molecular weight (reasonable range)
            mw = analysis.molecular_weight()
            mw_score = 1.0 if 5000 <= mw <= 100000 else 0.5
            
            # Check charge distribution
            charge = analysis.charge_at_pH(7.0)
            charge_score = 1.0 if abs(charge) <= len(sequence) * 0.2 else 0.7
            
            # Check hydrophobicity
            hydrophobicity = analysis.gravy()
            hydro_score = 1.0 if -2.0 <= hydrophobicity <= 2.0 else 0.7
            
            return (mw_score + charge_score + hydro_score) / 3.0
            
        except Exception:
            return 0.5
    
    def _evaluate_structural_similarity(self, sequence: str, 
                                      structure_tokens: str) -> float:
        """Evaluate similarity to reference structure if available."""
        # This would require converting structure tokens to coordinates
        # and computing RMSD/TM-score with reference
        # For now, return neutral score
        return 0.5


class InverseFoldingEvaluator(BaseEvaluator):
    """
    Evaluator for inverse folding task.
    
    Given structure and designed sequence, evaluates:
    - Sequence-structure compatibility (how well sequence fits structure)
    - Amino acid recovery (if reference sequence available)
    - Biophysical properties
    - Designability metrics
    """
    
    def __init__(self, reference_sequence: Optional[str] = None,
                 device: str = "cuda"):
        """
        Initialize inverse folding evaluator.
        
        Args:
            reference_sequence: Reference sequence for AAR calculation (optional)
            device: CUDA device
        """
        super().__init__(device)
        self.reference_sequence = reference_sequence
        
        # Load ESMFold for structure prediction if available
        self.esmfold_model = None
        if ESM_AVAILABLE:
            try:
                self.esmfold_model = esm.pretrained.esmfold_v1()
                self.esmfold_model = self.esmfold_model.eval().to(device)
                logger.info("✅ ESMFold loaded for sequence validation")
            except Exception as e:
                logger.warning(f"Failed to load ESMFold: {e}")
    
    def evaluate_sequence(self, sequence: str, structure_tokens: str) -> float:
        """
        Evaluate sequence design quality.
        
        Args:
            sequence: Designed amino acid sequence
            structure_tokens: Target structure tokens
            
        Returns:
            Reward score (higher is better)
        """
        if not sequence or not structure_tokens:
            return 0.0
        
        # Create cache key
        cache_key = f"inv_fold_{hash(sequence)}_{hash(structure_tokens)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        reward = 0.0
        
        try:
            # 1. Amino acid recovery (if reference available)
            if self.reference_sequence:
                aar_score = self._calculate_aar(sequence, self.reference_sequence)
                reward += aar_score * 0.4
            
            # 2. Sequence validity and naturalness
            naturalness_score = self._evaluate_sequence_naturalness(sequence)
            reward += naturalness_score * 0.3
            
            # 3. Structure compatibility (fold sequence and compare)
            if self.esmfold_model is not None:
                compatibility_score = self._evaluate_fold_compatibility(
                    sequence, structure_tokens
                )
                reward += compatibility_score * 0.2
            
            # 4. Biophysical properties
            biophysical_score = self._evaluate_biophysical_properties(sequence)
            reward += biophysical_score * 0.1
            
            # If no reference sequence, give more weight to other metrics
            if not self.reference_sequence:
                reward = (naturalness_score * 0.5 + 
                         biophysical_score * 0.3 + 
                         (compatibility_score if self.esmfold_model else 0.5) * 0.2)
            
            # Normalize reward to [0, 1]
            reward = max(0.0, min(1.0, reward))
            
        except Exception as e:
            logger.warning(f"Sequence evaluation failed: {e}")
            reward = 0.0
        
        # Cache result
        self.cache[cache_key] = reward
        return reward
    
    def _calculate_aar(self, predicted_seq: str, reference_seq: str) -> float:
        """Calculate Amino Acid Recovery (AAR)."""
        if len(predicted_seq) != len(reference_seq):
            # Align to shorter sequence
            min_len = min(len(predicted_seq), len(reference_seq))
            predicted_seq = predicted_seq[:min_len]
            reference_seq = reference_seq[:min_len]
        
        if len(reference_seq) == 0:
            return 0.0
        
        matches = sum(1 for p, r in zip(predicted_seq, reference_seq) if p == r)
        return matches / len(reference_seq)
    
    def _evaluate_sequence_naturalness(self, sequence: str) -> float:
        """Evaluate if sequence looks natural/realistic."""
        try:
            # Check for reasonable length
            if len(sequence) < 5:
                return 0.1
            
            # Check amino acid composition
            aa_counts = {}
            for aa in sequence:
                if aa not in "ACDEFGHIKLMNPQRSTVWY":
                    return 0.1  # Invalid amino acid
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
            # Penalize if dominated by single amino acid
            max_freq = max(aa_counts.values()) / len(sequence)
            if max_freq > 0.6:
                return 0.3
            
            # Check for natural amino acid distribution
            # Common amino acids should appear reasonably
            common_aas = set("AGLVISE")  # Common amino acids
            common_count = sum(aa_counts.get(aa, 0) for aa in common_aas)
            common_ratio = common_count / len(sequence)
            
            naturalness = 0.5 + 0.5 * min(1.0, common_ratio / 0.4)
            return naturalness
            
        except Exception:
            return 0.3
    
    def _evaluate_fold_compatibility(self, sequence: str, structure_tokens: str) -> float:
        """Evaluate how well sequence folds to match target structure."""
        try:
            if self.esmfold_model is None:
                return 0.5
            
            with torch.no_grad():
                # Predict structure for designed sequence
                output = self.esmfold_model.infer_pdb(sequence)
                
                # Extract confidence scores (pLDDT)
                if hasattr(output, 'plddt'):
                    plddt_scores = output.plddt
                    mean_plddt = float(plddt_scores.mean())
                    
                    # High pLDDT indicates good foldability
                    fold_score = mean_plddt / 100.0
                    
                    # TODO: Compare predicted structure to target structure tokens
                    # For now, just use pLDDT as proxy for compatibility
                    return fold_score
                
                return 0.6  # Neutral score if no pLDDT available
                
        except Exception as e:
            logger.debug(f"Fold compatibility evaluation failed: {e}")
            return 0.5
    
    def _evaluate_biophysical_properties(self, sequence: str) -> float:
        """Evaluate biophysical properties of the sequence."""
        try:
            if not BIO_AVAILABLE:
                return 0.5
            
            analysis = ProteinAnalysis(sequence)
            
            # Check isoelectric point
            iep = analysis.isoelectric_point()
            iep_score = 1.0 if 4.0 <= iep <= 10.0 else 0.7
            
            # Check instability index
            instability = analysis.instability_index()
            stability_score = 1.0 if instability < 40 else 0.6
            
            # Check secondary structure fraction
            ss_frac = analysis.secondary_structure_fraction()
            # Reasonable proteins have some secondary structure
            ss_score = min(1.0, (ss_frac[0] + ss_frac[1]) / 0.3)  # helix + sheet
            
            return (iep_score + stability_score + ss_score) / 3.0
            
        except Exception:
            return 0.5


class MotifScaffoldingEvaluator(BaseEvaluator):
    """
    Evaluator for motif scaffolding task.
    
    Given partial structure with motif and completed scaffold,
    evaluates scaffold quality and motif preservation.
    """
    
    def __init__(self, motif_positions: List[int], device: str = "cuda"):
        """
        Initialize motif scaffolding evaluator.
        
        Args:
            motif_positions: List of positions that define the motif
            device: CUDA device
        """
        super().__init__(device)
        self.motif_positions = motif_positions
    
    def evaluate_scaffold(self, sequence: str, structure_tokens: str,
                         motif_sequence: str, motif_structure: str) -> float:
        """
        Evaluate scaffold completion quality.
        
        Args:
            sequence: Complete sequence with motif + scaffold
            structure_tokens: Complete structure tokens
            motif_sequence: Original motif sequence
            motif_structure: Original motif structure
            
        Returns:
            Reward score (higher is better)
        """
        # TODO: Implement motif scaffolding evaluation
        # This is more complex and requires:
        # 1. Motif preservation check
        # 2. Scaffold-motif interface quality
        # 3. Overall structure stability
        # 4. Designability of scaffold regions
        
        return 0.5  # Placeholder


def create_evaluator(task_type: str, **kwargs) -> BaseEvaluator:
    """
    Factory function to create appropriate evaluator for task type.
    
    Args:
        task_type: "folding", "inverse_folding", or "motif_scaffolding"
        **kwargs: Task-specific arguments
        
    Returns:
        Appropriate evaluator instance
    """
    if task_type == "folding":
        return FoldingEvaluator(**kwargs)
    elif task_type == "inverse_folding":
        return InverseFoldingEvaluator(**kwargs)
    elif task_type == "motif_scaffolding":
        return MotifScaffoldingEvaluator(**kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

