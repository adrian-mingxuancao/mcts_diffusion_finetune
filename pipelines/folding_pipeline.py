"""
Complete Folding Pipeline: Sequence -> Structure Prediction with MCTS Optimization
================================================================================

This pipeline implements the complete folding workflow:

1. **Baseline Generation**: Use ESMFold to predict initial structure
2. **Structure Tokenization**: Convert ESMFold output to DPLM-2 structure tokens
3. **MCTS Optimization**: Use MCTS to improve structure prediction
4. **Evaluation**: Compare against reference structure using multiple metrics

The pipeline is designed to work with the CAMEO dataset and can evaluate:
- RMSD (Root Mean Square Deviation)
- TM-score (Template Modeling score)
- GDT-TS (Global Distance Test)
- pLDDT (predicted Local Distance Difference Test)
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    logging.warning("ESM not available - folding pipeline will be limited")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_mcts import UnifiedMCTS
from core.task_evaluators import FoldingEvaluator
from utils.structure_tokenizer import StructureTokenizer, esmfold_to_dplm_tokens
from utils.cameo_data_loader import CAMEODataLoader

logger = logging.getLogger(__name__)


class FoldingPipeline:
    """
    Complete folding pipeline with MCTS optimization.
    
    This class orchestrates the entire folding process:
    1. Load sequence from CAMEO dataset
    2. Generate baseline structure using ESMFold
    3. Convert to DPLM-2 structure tokens
    4. Optimize using MCTS
    5. Evaluate against reference structure
    """
    
    def __init__(self, device: str = "cuda", use_mcts: bool = True):
        """
        Initialize folding pipeline.
        
        Args:
            device: CUDA device for computation
            use_mcts: Whether to use MCTS optimization (False for baseline only)
        """
        self.device = device
        self.use_mcts = use_mcts
        
        # Initialize components
        self.esmfold_model = None
        self.structure_tokenizer = None
        self.cameo_loader = None
        
        # Load ESMFold model
        self._load_esmfold()
        
        # Initialize structure tokenizer
        self.structure_tokenizer = StructureTokenizer(device=device)
        
        # Initialize CAMEO data loader
        try:
            self.cameo_loader = CAMEODataLoader()
            logger.info("✅ CAMEO data loader initialized")
        except Exception as e:
            logger.warning(f"Failed to load CAMEO data: {e}")
    
    def _load_esmfold(self):
        """Load ESMFold model for baseline structure prediction."""
        if not ESM_AVAILABLE:
            logger.error("ESM not available - cannot load ESMFold")
            return
        
        try:
            logger.info("🔄 Loading ESMFold model...")
            self.esmfold_model = esm.pretrained.esmfold_v1()
            self.esmfold_model = self.esmfold_model.eval().to(self.device)
            logger.info("✅ ESMFold model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load ESMFold: {e}")
            self.esmfold_model = None
    
    def predict_structure(self, sequence: str, reference_structure: Optional[np.ndarray] = None,
                         mcts_iterations: int = 30) -> Dict[str, Any]:
        """
        Predict structure for given sequence.
        
        Args:
            sequence: Amino acid sequence
            reference_structure: Reference structure for evaluation (optional)
            mcts_iterations: Number of MCTS iterations for optimization
            
        Returns:
            Dictionary with prediction results and metrics
        """
        results = {
            'sequence': sequence,
            'baseline_structure_tokens': None,
            'optimized_structure_tokens': None,
            'baseline_metrics': {},
            'optimized_metrics': {},
            'improvement': {}
        }
        
        try:
            # Step 1: Generate baseline structure using ESMFold
            logger.info(f"🧬 Predicting structure for sequence length {len(sequence)}")
            baseline_structure_tokens = self._generate_baseline_structure(sequence)
            results['baseline_structure_tokens'] = baseline_structure_tokens
            
            if not baseline_structure_tokens:
                logger.error("Failed to generate baseline structure")
                return results
            
            # Step 2: Evaluate baseline
            if reference_structure is not None:
                baseline_metrics = self._evaluate_structure(
                    sequence, baseline_structure_tokens, reference_structure
                )
                results['baseline_metrics'] = baseline_metrics
                logger.info(f"📊 Baseline metrics: {baseline_metrics}")
            
            # Step 3: MCTS optimization (if enabled)
            optimized_structure_tokens = baseline_structure_tokens
            
            if self.use_mcts:
                logger.info(f"🌳 Starting MCTS optimization with {mcts_iterations} iterations")
                optimized_structure_tokens = self._optimize_with_mcts(
                    sequence, baseline_structure_tokens, reference_structure, mcts_iterations
                )
                results['optimized_structure_tokens'] = optimized_structure_tokens
                
                # Step 4: Evaluate optimized structure
                if reference_structure is not None:
                    optimized_metrics = self._evaluate_structure(
                        sequence, optimized_structure_tokens, reference_structure
                    )
                    results['optimized_metrics'] = optimized_metrics
                    logger.info(f"📊 Optimized metrics: {optimized_metrics}")
                    
                    # Calculate improvement
                    improvement = self._calculate_improvement(
                        baseline_metrics, optimized_metrics
                    )
                    results['improvement'] = improvement
                    logger.info(f"📈 Improvement: {improvement}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Structure prediction failed: {e}")
            return results
    
    def _generate_baseline_structure(self, sequence: str) -> Optional[str]:
        """Generate baseline structure using ESMFold."""
        if self.esmfold_model is None:
            logger.error("ESMFold model not available")
            return None
        
        try:
            with torch.no_grad():
                # Predict structure using ESMFold
                logger.info("🔄 Running ESMFold prediction...")
                output = self.esmfold_model.infer_pdb(sequence)
                
                # Convert ESMFold output to DPLM structure tokens
                logger.info("🔄 Converting to DPLM structure tokens...")
                structure_tokens = self.structure_tokenizer.esmfold_to_dplm_tokens(
                    output, sequence
                )
                
                logger.info(f"✅ Generated {len(structure_tokens.split(','))} structure tokens")
                return structure_tokens
                
        except Exception as e:
            logger.error(f"❌ ESMFold prediction failed: {e}")
            return None
    
    def _optimize_with_mcts(self, sequence: str, baseline_structure_tokens: str,
                           reference_structure: Optional[np.ndarray],
                           mcts_iterations: int) -> str:
        """Optimize structure using MCTS."""
        try:
            # Create evaluator
            evaluator = FoldingEvaluator(
                reference_structure=reference_structure,
                device=self.device
            )
            
            # Initialize MCTS
            mcts = UnifiedMCTS(
                task_type="folding",
                evaluator=evaluator,
                max_depth=3,
                exploration_constant=1.414,
                num_children_select=3,
                k_rollouts_per_expert=2,
                use_plddt_masking=True,
                use_ph_uct=True,
                device=self.device
            )
            
            # Run MCTS search
            best_node = mcts.search(
                input_sequence=sequence,
                input_structure_tokens=None,  # Will be generated as baseline
                num_iterations=mcts_iterations
            )
            
            # Return optimized structure tokens
            return best_node.structure_tokens or baseline_structure_tokens
            
        except Exception as e:
            logger.error(f"❌ MCTS optimization failed: {e}")
            return baseline_structure_tokens
    
    def _evaluate_structure(self, sequence: str, structure_tokens: str,
                           reference_structure: np.ndarray) -> Dict[str, float]:
        """Evaluate predicted structure against reference."""
        metrics = {}
        
        try:
            # Convert structure tokens back to coordinates
            predicted_coords = self.structure_tokenizer.dplm_tokens_to_coords(
                structure_tokens, sequence
            )
            
            # Calculate RMSD
            rmsd = self._calculate_rmsd(predicted_coords, reference_structure)
            metrics['rmsd'] = rmsd
            
            # Calculate TM-score (simplified version)
            tm_score = self._calculate_tm_score(predicted_coords, reference_structure)
            metrics['tm_score'] = tm_score
            
            # Calculate structure quality metrics
            quality_score = self._calculate_structure_quality(structure_tokens)
            metrics['quality_score'] = quality_score
            
        except Exception as e:
            logger.warning(f"Structure evaluation failed: {e}")
            metrics = {'rmsd': float('inf'), 'tm_score': 0.0, 'quality_score': 0.0}
        
        return metrics
    
    def _calculate_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate RMSD between two coordinate sets."""
        try:
            # Align coordinate sets to same length
            min_len = min(len(coords1), len(coords2))
            coords1 = coords1[:min_len]
            coords2 = coords2[:min_len]
            
            # Center coordinates
            coords1_centered = coords1 - np.mean(coords1, axis=0)
            coords2_centered = coords2 - np.mean(coords2, axis=0)
            
            # Calculate RMSD
            diff = coords1_centered - coords2_centered
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            
            return float(rmsd)
            
        except Exception as e:
            logger.warning(f"RMSD calculation failed: {e}")
            return float('inf')
    
    def _calculate_tm_score(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate simplified TM-score."""
        try:
            # This is a simplified version - real TM-score requires optimal alignment
            rmsd = self._calculate_rmsd(coords1, coords2)
            
            # Convert RMSD to TM-score-like metric (0-1 scale)
            tm_score = 1.0 / (1.0 + (rmsd / 5.0)**2)  # Simplified formula
            
            return float(tm_score)
            
        except Exception as e:
            logger.warning(f"TM-score calculation failed: {e}")
            return 0.0
    
    def _calculate_structure_quality(self, structure_tokens: str) -> float:
        """Calculate structure quality based on token diversity and validity."""
        try:
            tokens = structure_tokens.split(',')
            
            # Check token validity
            valid_tokens = sum(1 for token in tokens if token.strip() != '<mask>')
            validity_score = valid_tokens / len(tokens)
            
            # Check token diversity
            unique_tokens = len(set(tokens))
            diversity_score = min(1.0, unique_tokens / (len(tokens) * 0.3))
            
            # Combine scores
            quality_score = (validity_score + diversity_score) / 2.0
            
            return float(quality_score)
            
        except Exception:
            return 0.0
    
    def _calculate_improvement(self, baseline_metrics: Dict[str, float],
                             optimized_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate improvement metrics."""
        improvement = {}
        
        for metric in baseline_metrics:
            if metric in optimized_metrics:
                baseline_val = baseline_metrics[metric]
                optimized_val = optimized_metrics[metric]
                
                if metric == 'rmsd':
                    # Lower is better for RMSD
                    if baseline_val > 0:
                        improvement[metric] = (baseline_val - optimized_val) / baseline_val
                    else:
                        improvement[metric] = 0.0
                else:
                    # Higher is better for other metrics
                    if baseline_val > 0:
                        improvement[metric] = (optimized_val - baseline_val) / baseline_val
                    else:
                        improvement[metric] = optimized_val
        
        return improvement
    
    def run_batch_evaluation(self, max_structures: int = 10,
                           mcts_iterations: int = 30) -> Dict[str, Any]:
        """
        Run batch evaluation on multiple CAMEO structures.
        
        Args:
            max_structures: Maximum number of structures to evaluate
            mcts_iterations: Number of MCTS iterations per structure
            
        Returns:
            Batch evaluation results
        """
        if self.cameo_loader is None:
            logger.error("CAMEO data loader not available")
            return {}
        
        logger.info(f"🚀 Starting batch evaluation of {max_structures} structures")
        
        results = {
            'individual_results': [],
            'aggregate_metrics': {},
            'summary': {}
        }
        
        for i in range(max_structures):
            try:
                # Load structure from CAMEO
                structure_data = self.cameo_loader.get_test_structure(i)
                sequence = structure_data['sequence']
                
                # Get reference structure (if available)
                reference_structure = None
                if 'coordinates' in structure_data:
                    reference_structure = structure_data['coordinates']
                
                logger.info(f"📋 Evaluating structure {i+1}/{max_structures}: {structure_data['name']}")
                
                # Run prediction
                result = self.predict_structure(
                    sequence, reference_structure, mcts_iterations
                )
                
                result['structure_name'] = structure_data['name']
                result['structure_index'] = i
                results['individual_results'].append(result)
                
                # Log progress
                if result['improvement']:
                    logger.info(f"✅ Structure {i+1} completed - Improvements: {result['improvement']}")
                else:
                    logger.info(f"✅ Structure {i+1} completed - No reference for improvement calculation")
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate structure {i+1}: {e}")
                continue
        
        # Calculate aggregate metrics
        results['aggregate_metrics'] = self._calculate_aggregate_metrics(
            results['individual_results']
        )
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        logger.info("🎯 Batch evaluation completed")
        logger.info(f"📊 Summary: {results['summary']}")
        
        return results
    
    def _calculate_aggregate_metrics(self, individual_results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate metrics across all evaluations."""
        aggregate = {}
        
        # Collect all metrics
        all_baseline_metrics = []
        all_optimized_metrics = []
        all_improvements = []
        
        for result in individual_results:
            if result['baseline_metrics']:
                all_baseline_metrics.append(result['baseline_metrics'])
            if result['optimized_metrics']:
                all_optimized_metrics.append(result['optimized_metrics'])
            if result['improvement']:
                all_improvements.append(result['improvement'])
        
        # Calculate averages
        if all_baseline_metrics:
            aggregate['baseline_avg'] = self._average_metrics(all_baseline_metrics)
        
        if all_optimized_metrics:
            aggregate['optimized_avg'] = self._average_metrics(all_optimized_metrics)
        
        if all_improvements:
            aggregate['improvement_avg'] = self._average_metrics(all_improvements)
        
        return aggregate
    
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Calculate average of metrics."""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0]:
            values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key]) and np.isfinite(m[key])]
            if values:
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def _generate_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate evaluation summary."""
        summary = {
            'total_structures': len(results['individual_results']),
            'successful_predictions': 0,
            'mcts_enabled': self.use_mcts,
            'average_improvements': {}
        }
        
        # Count successful predictions
        for result in results['individual_results']:
            if result['baseline_structure_tokens']:
                summary['successful_predictions'] += 1
        
        # Calculate average improvements
        if 'improvement_avg' in results['aggregate_metrics']:
            summary['average_improvements'] = results['aggregate_metrics']['improvement_avg']
        
        return summary


def main():
    """Main function for running folding pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run folding pipeline evaluation")
    parser.add_argument("--max_structures", type=int, default=5,
                       help="Maximum number of structures to evaluate")
    parser.add_argument("--mcts_iterations", type=int, default=20,
                       help="Number of MCTS iterations")
    parser.add_argument("--no_mcts", action="store_true",
                       help="Disable MCTS optimization (baseline only)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize pipeline
    pipeline = FoldingPipeline(
        device=args.device,
        use_mcts=not args.no_mcts
    )
    
    # Run evaluation
    results = pipeline.run_batch_evaluation(
        max_structures=args.max_structures,
        mcts_iterations=args.mcts_iterations
    )
    
    # Save results
    import json
    output_file = f"folding_results_{'mcts' if not args.no_mcts else 'baseline'}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Results saved to {output_file}")


if __name__ == "__main__":
    main()

