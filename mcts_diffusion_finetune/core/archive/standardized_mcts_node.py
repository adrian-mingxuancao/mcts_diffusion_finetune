"""
Standardized MCTS Node Structure
Supports both DPLM-2 (sequence + structure tokens) and external models (sequence + coordinates)
"""

import numpy as np
import torch
from typing import Dict, List, Set, Optional, Union, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class StandardizedMCTSNode:
    """
    Standardized MCTS Node that works with both DPLM-2 and external models
    
    Stores both:
    - DPLM-2 format: sequence + structure_tokens
    - External format: sequence + coordinates
    """
    # Core sequence information
    sequence: str
    
    # Structure information (multiple formats for compatibility)
    structure_tokens: Optional[str] = None  # DPLM-2 format: "159,162,163,164,..."
    coordinates: Optional[np.ndarray] = None  # External models format: (N, 3) or (N, 37, 3)
    
    # MCTS tree information
    parent: Optional['StandardizedMCTSNode'] = None
    children: List['StandardizedMCTSNode'] = field(default_factory=list)
    depth: int = 0
    
    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0
    reward: float = 0.0
    
    # pH-UCT bonuses (cached at expansion)
    entropy_bonus: float = 0.0
    diversity_bonus: float = 0.0
    
    # Masking information for MCTS
    masked_positions: Set[int] = field(default_factory=set)
    
    # Generation metadata
    generated_by: str = "unknown"  # Which expert generated this node
    generation_entropy: float = 1.0
    
    @property
    def average_value(self) -> float:
        """Q-value: average reward from this node"""
        return self.total_value / max(self.visit_count, 1)
    
    def ph_uct_score(self, exploration_constant: float = 1.414, w_ent: float = 0.1, w_div: float = 0.1) -> float:
        """
        pH-UCT-ME score with correct multiplication formula:
        Q + c_p * sqrt(log(N(s)) / (1 + N(s,a))) * (w_ent * U_ent + w_div * U_div)
        """
        if self.visit_count == 0:
            return float('inf')  # Prioritize unvisited nodes
        
        parent_visits = self.parent.visit_count if self.parent else 1
        
        # Q-value (exploitation)
        q_value = self.average_value
        
        # UCB exploration base
        ucb_base = np.sqrt(np.log(parent_visits) / (1 + self.visit_count))
        
        # pH-UCT-ME bonus (cached at expansion)
        ph_me_bonus = w_ent * self.entropy_bonus + w_div * self.diversity_bonus
        
        # Final pH-UCT-ME score with multiplication
        ph_uct_score = q_value + exploration_constant * ucb_base * ph_me_bonus
        
        return ph_uct_score
    
    def has_structure_tokens(self) -> bool:
        """Check if node has DPLM-2 structure tokens"""
        return self.structure_tokens is not None and len(self.structure_tokens) > 0
    
    def has_coordinates(self) -> bool:
        """Check if node has coordinate information"""
        return self.coordinates is not None and self.coordinates.size > 0
    
    def get_coordinates_for_external_models(self) -> Optional[np.ndarray]:
        """Get coordinates in format suitable for external models"""
        if self.has_coordinates():
            return self.coordinates
        
        # If we only have structure tokens, we'd need to convert them to coordinates
        # This would require ESMFold or similar structure prediction
        logger.warning("Node has structure tokens but no coordinates - conversion needed")
        return None
    
    def get_structure_tokens_for_dplm2(self) -> Optional[str]:
        """Get structure tokens in format suitable for DPLM-2"""
        if self.has_structure_tokens():
            return self.structure_tokens
        
        # If we only have coordinates, we'd need to convert them to structure tokens
        # This would require structure tokenization
        logger.warning("Node has coordinates but no structure tokens - conversion needed")
        return None
    
    def update_structure_from_coordinates(self, coordinates: np.ndarray, method: str = "external"):
        """Update node structure information from coordinates"""
        self.coordinates = coordinates
        self.generated_by = method
        
        # TODO: Convert coordinates to structure tokens for DPLM-2 compatibility
        # For now, create placeholder tokens
        if coordinates is not None:
            seq_len = len(self.sequence)
            self.structure_tokens = ",".join(["0000"] * seq_len)
    
    def update_structure_from_tokens(self, structure_tokens: str, method: str = "dplm2"):
        """Update node structure information from DPLM-2 tokens"""
        self.structure_tokens = structure_tokens
        self.generated_by = method
        
        # TODO: Convert structure tokens to coordinates for external model compatibility
        # For now, coordinates remain None until we predict structure
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        return {
            "sequence": self.sequence,
            "structure_tokens": self.structure_tokens,
            "coordinates": self.coordinates.tolist() if self.coordinates is not None else None,
            "depth": self.depth,
            "visit_count": self.visit_count,
            "total_value": self.total_value,
            "reward": self.reward,
            "entropy_bonus": self.entropy_bonus,
            "diversity_bonus": self.diversity_bonus,
            "generated_by": self.generated_by,
            "generation_entropy": self.generation_entropy,
            "masked_positions": list(self.masked_positions)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardizedMCTSNode':
        """Create node from dictionary"""
        coordinates = None
        if data.get("coordinates") is not None:
            coordinates = np.array(data["coordinates"])
        
        return cls(
            sequence=data["sequence"],
            structure_tokens=data.get("structure_tokens"),
            coordinates=coordinates,
            depth=data.get("depth", 0),
            visit_count=data.get("visit_count", 0),
            total_value=data.get("total_value", 0.0),
            reward=data.get("reward", 0.0),
            entropy_bonus=data.get("entropy_bonus", 0.0),
            diversity_bonus=data.get("diversity_bonus", 0.0),
            generated_by=data.get("generated_by", "unknown"),
            generation_entropy=data.get("generation_entropy", 1.0),
            masked_positions=set(data.get("masked_positions", []))
        )


class StructureConverter:
    """Converts between different structure representations"""
    
    @staticmethod
    def tokens_to_coordinates(structure_tokens: str, sequence: str) -> Optional[np.ndarray]:
        """Convert DPLM-2 structure tokens to coordinates (requires structure prediction)"""
        # This would use ESMFold or similar to predict coordinates from sequence
        # For now, return None to indicate conversion needed
        logger.warning("Structure token to coordinate conversion not implemented")
        return None
    
    @staticmethod
    def coordinates_to_tokens(coordinates: np.ndarray, sequence: str) -> Optional[str]:
        """Convert coordinates to DPLM-2 structure tokens (requires structure tokenization)"""
        # This would use the DPLM-2 structure tokenizer to convert coordinates to tokens
        # For now, create placeholder tokens
        seq_len = len(sequence)
        placeholder_tokens = ",".join(["0000"] * seq_len)
        logger.warning("Coordinate to structure token conversion not implemented - using placeholders")
        return placeholder_tokens
    
    @staticmethod
    def ensure_coordinates(node: StandardizedMCTSNode, esmfold_model=None) -> np.ndarray:
        """Ensure node has coordinates, predict if necessary"""
        if node.has_coordinates():
            return node.coordinates
        
        # Predict coordinates from sequence using ESMFold
        if esmfold_model is not None:
            try:
                # Use ESMFold to predict structure
                logger.info(f"Predicting coordinates for sequence: {node.sequence[:20]}...")
                
                # This would be the actual ESMFold prediction
                # For now, return mock coordinates
                seq_len = len(node.sequence)
                mock_coords = np.random.rand(seq_len, 3) * 50  # Mock protein-sized coordinates
                
                node.coordinates = mock_coords
                return mock_coords
                
            except Exception as e:
                logger.error(f"Failed to predict coordinates: {e}")
                return None
        
        logger.warning("No ESMFold model available for coordinate prediction")
        return None
    
    @staticmethod
    def ensure_structure_tokens(node: StandardizedMCTSNode, structure_tokenizer=None) -> str:
        """Ensure node has structure tokens, tokenize if necessary"""
        if node.has_structure_tokens():
            return node.structure_tokens
        
        # Convert coordinates to structure tokens
        if node.has_coordinates() and structure_tokenizer is not None:
            try:
                # This would be the actual structure tokenization
                # For now, return placeholder tokens
                seq_len = len(node.sequence)
                placeholder_tokens = ",".join(["0000"] * seq_len)
                
                node.structure_tokens = placeholder_tokens
                return placeholder_tokens
                
            except Exception as e:
                logger.error(f"Failed to tokenize structure: {e}")
                return ""
        
        logger.warning("No coordinates or structure tokenizer available")
        return ""


def test_standardized_node():
    """Test the standardized node structure"""
    print("🧪 Testing Standardized MCTS Node")
    print("=" * 40)
    
    # Test DPLM-2 style node
    print("🔧 Testing DPLM-2 style node...")
    dplm2_node = StandardizedMCTSNode(
        sequence="MQIFAAAAAAAAAA",
        structure_tokens="159,162,163,164,0000,0000,0000,0000,0000,0000,0000,0000,0000,0000",
        generated_by="dplm2"
    )
    
    print(f"   ✅ DPLM-2 node: {dplm2_node.sequence}")
    print(f"   🏗️ Has structure tokens: {dplm2_node.has_structure_tokens()}")
    print(f"   📊 Has coordinates: {dplm2_node.has_coordinates()}")
    
    # Test external model style node
    print("🔧 Testing external model style node...")
    mock_coords = np.random.rand(14, 3) * 50
    external_node = StandardizedMCTSNode(
        sequence="MQIFBBBBBBBBBB",
        coordinates=mock_coords,
        generated_by="foldflow"
    )
    
    print(f"   ✅ External node: {external_node.sequence}")
    print(f"   🏗️ Has structure tokens: {external_node.has_structure_tokens()}")
    print(f"   📊 Has coordinates: {external_node.has_coordinates()}")
    print(f"   📐 Coordinates shape: {external_node.coordinates.shape}")
    
    # Test conversion
    print("🔧 Testing structure conversion...")
    converter = StructureConverter()
    
    # Convert external node to have structure tokens
    tokens = converter.ensure_structure_tokens(external_node)
    print(f"   🏗️ Generated tokens: {tokens[:50]}...")
    
    # Convert DPLM-2 node to have coordinates
    coords = converter.ensure_coordinates(dplm2_node)
    if coords is not None:
        print(f"   📊 Generated coordinates: {coords.shape}")
    
    print("🎉 Standardized node testing completed!")
    return True

if __name__ == "__main__":
    test_standardized_node()





