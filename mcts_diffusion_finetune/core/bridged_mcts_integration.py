"""
Bridged MCTS Integration for DPLM-2 + External Models
Uses environment bridge to integrate external models with existing DPLM-2 MCTS
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from external_models.environment_bridge import BridgedExternalExperts
from core.motif_scaffolding_mcts import MotifScaffoldingMCTS, MotifScaffoldingData

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BridgedMotifScaffoldingMCTS(MotifScaffoldingMCTS):
    """MCTS for motif scaffolding with bridged external experts"""
    
    def __init__(self, dplm2_integration, use_external_experts: bool = True):
        """
        Initialize MCTS with bridged external experts
        
        Args:
            dplm2_integration: DPLM2Integration instance
            use_external_experts: Whether to use external experts via bridge
        """
        # Initialize base MCTS
        super().__init__(dplm2_integration, external_experts=None)
        
        # Initialize bridged external experts
        self.use_external_experts = use_external_experts
        if use_external_experts:
            try:
                self.bridged_experts = BridgedExternalExperts()
                self.available_external_experts = self.bridged_experts.get_available_expert_names()
                logger.info(f"✅ Bridged external experts initialized: {self.available_external_experts}")
            except Exception as e:
                logger.error(f"Failed to initialize bridged experts: {e}")
                self.bridged_experts = None
                self.available_external_experts = []
        else:
            self.bridged_experts = None
            self.available_external_experts = []
    
    def _expert_rollout_with_masking(self, expert_name: str, motif_data: MotifScaffoldingData, 
                                   masked_sequence: str, masked_positions: set, 
                                   structure_tokens: Optional[List[str]] = None) -> tuple:
        """Expert rollout using bridged external models"""
        
        if not self.use_external_experts or not self.bridged_experts:
            # Fallback to DPLM-2 only
            return self._dplm2_rollout_with_masking(motif_data, masked_sequence, masked_positions, structure_tokens)
        
        if expert_name not in self.available_external_experts:
            logger.warning(f"Expert '{expert_name}' not available, using DPLM-2")
            return self._dplm2_rollout_with_masking(motif_data, masked_sequence, masked_positions, structure_tokens)
        
        try:
            logger.info(f"🔧 Running {expert_name} via bridge...")
            
            if expert_name == "proteina":
                # ProteInA motif scaffolding
                results = self.bridged_experts.expert_rollout(
                    expert_name,
                    motif_pdb=motif_data.motif_pdb,
                    scaffold_length=motif_data.target_length,
                    num_samples=1
                )
                
                if results:
                    # Parse PDB to get sequence (simplified)
                    pdb_content = results[0]
                    # Extract sequence from PDB (this is a simplified version)
                    sequence = self._extract_sequence_from_pdb(pdb_content, motif_data.target_length)
                    
                    # Generate structure tokens (placeholder)
                    new_structure_tokens = ["0000"] * motif_data.target_length
                    
                    logger.info(f"✅ {expert_name} generated sequence: {sequence[:20]}...")
                    return sequence, new_structure_tokens
                else:
                    logger.warning(f"{expert_name} returned no results")
            
            elif expert_name == "foldflow":
                # FoldFlow structure generation
                results = self.bridged_experts.expert_rollout(
                    expert_name,
                    length=motif_data.target_length,
                    num_samples=1
                )
                
                if results:
                    # Parse PDB to get sequence (simplified)
                    pdb_content = results[0]
                    sequence = self._extract_sequence_from_pdb(pdb_content, motif_data.target_length)
                    
                    # Generate structure tokens (placeholder)
                    new_structure_tokens = ["0000"] * motif_data.target_length
                    
                    logger.info(f"✅ {expert_name} generated sequence: {sequence[:20]}...")
                    return sequence, new_structure_tokens
                else:
                    logger.warning(f"{expert_name} returned no results")
            
            elif expert_name == "rfdiffusion":
                # RFDiffusion motif scaffolding
                results = self.bridged_experts.expert_rollout(
                    expert_name,
                    motif_pdb=motif_data.motif_pdb,
                    scaffold_length=motif_data.target_length,
                    num_samples=1
                )
                
                if results:
                    # Parse PDB to get sequence (simplified)
                    pdb_content = results[0]
                    sequence = self._extract_sequence_from_pdb(pdb_content, motif_data.target_length)
                    
                    # Generate structure tokens (placeholder)
                    new_structure_tokens = ["0000"] * motif_data.target_length
                    
                    logger.info(f"✅ {expert_name} generated sequence: {sequence[:20]}...")
                    return sequence, new_structure_tokens
                else:
                    logger.warning(f"{expert_name} returned no results")
            
            elif expert_name == "proteinmpnn":
                # ProteinMPNN sequence design from structure
                results = self.bridged_experts.expert_rollout(
                    expert_name,
                    pdb_content=motif_data.motif_pdb,
                    num_samples=1,
                    temperature=0.1
                )
                
                if results:
                    sequence = results[0]
                    # Pad or truncate to target length
                    if len(sequence) < motif_data.target_length:
                        sequence += "A" * (motif_data.target_length - len(sequence))
                    elif len(sequence) > motif_data.target_length:
                        sequence = sequence[:motif_data.target_length]
                    
                    # Generate structure tokens (placeholder)
                    new_structure_tokens = ["0000"] * motif_data.target_length
                    
                    logger.info(f"✅ {expert_name} generated sequence: {sequence[:20]}...")
                    return sequence, new_structure_tokens
                else:
                    logger.warning(f"{expert_name} returned no results")
            
            # If we get here, expert failed
            logger.warning(f"{expert_name} failed, falling back to DPLM-2")
            
        except Exception as e:
            logger.error(f"Expert {expert_name} failed: {e}")
        
        # Fallback to DPLM-2
        return self._dplm2_rollout_with_masking(motif_data, masked_sequence, masked_positions, structure_tokens)
    
    def _extract_sequence_from_pdb(self, pdb_content: str, target_length: int) -> str:
        """Extract amino acid sequence from PDB content (simplified)"""
        # This is a simplified version - in practice you'd use proper PDB parsing
        
        # Standard amino acid mapping
        aa_map = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
        sequence = ""
        seen_residues = set()
        
        for line in pdb_content.split('\n'):
            if line.startswith('ATOM') and ' CA ' in line:
                try:
                    # Parse residue info
                    res_name = line[17:20].strip()
                    res_num = int(line[22:26].strip())
                    
                    if res_num not in seen_residues:
                        aa = aa_map.get(res_name, 'A')  # Default to Alanine
                        sequence += aa
                        seen_residues.add(res_num)
                        
                except (ValueError, IndexError):
                    continue
        
        # Pad or truncate to target length
        if len(sequence) < target_length:
            sequence += "A" * (target_length - len(sequence))
        elif len(sequence) > target_length:
            sequence = sequence[:target_length]
        
        return sequence
    
    def search(self, motif_data: MotifScaffoldingData, num_iterations: int = 10, 
               expert_names: Optional[List[str]] = None) -> tuple:
        """
        Run MCTS search with bridged external experts
        
        Args:
            motif_data: Motif scaffolding data
            num_iterations: Number of MCTS iterations
            expert_names: List of expert names to use (None = use all available)
        
        Returns:
            (best_sequence, best_reward, search_stats)
        """
        
        # Determine which experts to use
        if expert_names is None:
            if self.use_external_experts and self.available_external_experts:
                expert_names = self.available_external_experts.copy()
            else:
                expert_names = []
        
        # Filter to available experts
        available_experts = [name for name in expert_names if name in self.available_external_experts]
        
        logger.info(f"🚀 Starting bridged MCTS search...")
        logger.info(f"   Motif: {motif_data.motif_sequence} ({len(motif_data.motif_positions)} residues)")
        logger.info(f"   Target length: {motif_data.target_length}")
        logger.info(f"   External experts: {available_experts}")
        logger.info(f"   MCTS iterations: {num_iterations}")
        
        # Run the search using parent class method
        return super().search(motif_data, num_iterations, available_experts)


def test_bridged_mcts():
    """Test the bridged MCTS integration"""
    print("🧪 Testing Bridged MCTS Integration")
    print("=" * 50)
    
    try:
        # Test motif data
        class TestMotifData:
            def __init__(self):
                self.motif_pdb = '''ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N  
ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 10.38           C  
ATOM      3  C   MET A   1      26.913  26.639   3.531  1.00  9.62           C  
ATOM      4  O   MET A   1      27.886  26.463   4.263  1.00  9.62           O  
END'''
                self.motif_sequence = "MQIF"
                self.motif_positions = [0, 1, 2, 3]
                self.target_length = 30
                self.scaffold_positions = list(range(4, 30))
        
        motif_data = TestMotifData()
        
        # Test without DPLM-2 (just external experts)
        print("🔧 Testing bridged external experts only...")
        
        # Mock DPLM-2 integration
        class MockDPLM2:
            def __init__(self):
                pass
        
        mock_dplm2 = MockDPLM2()
        
        # Initialize bridged MCTS
        mcts = BridgedMotifScaffoldingMCTS(mock_dplm2, use_external_experts=True)
        
        print(f"Available external experts: {mcts.available_external_experts}")
        
        if mcts.available_external_experts:
            print("✅ Bridged MCTS initialized successfully!")
            print("🚀 Ready for multi-expert motif scaffolding!")
        else:
            print("⚠️ No external experts available")
        
        return True
        
    except Exception as e:
        print(f"❌ Bridged MCTS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_bridged_mcts()





