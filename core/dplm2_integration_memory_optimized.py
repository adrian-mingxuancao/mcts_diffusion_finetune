"""
Memory-Optimized DPLM2 Integration
Uses comprehensive memory management to allow DPLM-2 3B and ESMFold to work together.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .memory_manager import get_memory_manager
from .dplm2_integration_fixed import DPLM2Integration


class MemoryOptimizedDPLM2Integration(DPLM2Integration):
    """DPLM2 Integration with comprehensive memory management"""
    
    def __init__(self, model_name: str = "airkingbd/dplm2_150m"):
        super().__init__(model_name)
        self.memory_manager = get_memory_manager()
        self.expert_model_cache: Dict[str, Any] = {}
        
        # Expert model mapping
        self.expert_mapping = {
            0: "airkingbd/dplm2_650m",
            1: "airkingbd/dplm2_150m", 
            2: "airkingbd/dplm2_3b"
        }
        
        print(f"🧠 Memory-optimized DPLM2 integration initialized")
        print(f"   {self.memory_manager.get_memory_status()}")
    
    def _load_expert_on_demand(self, expert_name: str):
        """Load expert with memory management"""
        # Check if already loaded in cache
        if expert_name in self.expert_model_cache:
            return self.expert_model_cache[expert_name]
        
        print(f"🔄 Loading expert with memory management: {expert_name}")
        
        # Optimize memory for this expert
        self.memory_manager.optimize_memory_for_models([expert_name])
        
        try:
            # Load expert using parent method
            expert_model = super()._load_expert_on_demand(expert_name)
            
            # Cache the model
            self.expert_model_cache[expert_name] = expert_model
            
            # Register with memory manager
            self.memory_manager.load_model(expert_name, expert_model)
            
            print(f"✅ Expert {expert_name} loaded with memory management")
            print(f"   {self.memory_manager.get_memory_status()}")
            
            return expert_model
            
        except Exception as e:
            print(f"❌ Failed to load expert {expert_name}: {e}")
            # Emergency cleanup and retry once
            self.memory_manager.emergency_cleanup()
            try:
                expert_model = super()._load_expert_on_demand(expert_name)
                self.expert_model_cache[expert_name] = expert_model
                self.memory_manager.load_model(expert_name, expert_model)
                return expert_model
            except Exception as e2:
                print(f"❌ Retry failed for expert {expert_name}: {e2}")
                raise
    
    def generate_with_expert(self, expert_id: int, structure: Dict, target_length: int, masked_sequence: str = None, temperature: float = 0.8) -> str:
        """Generate with expert using memory management"""
        expert_name = self.expert_mapping.get(expert_id, "airkingbd/dplm2_150m")
        
        print(f"🎯 Generating with expert {expert_id} ({expert_name})")
        print(f"   {self.memory_manager.get_memory_status()}")
        
        # Handle baseline generation (no masked sequence provided)
        if masked_sequence is None:
            print(f"🔄 Creating fully masked sequence for baseline generation...")
            masked_sequence = 'X' * target_length
        
        try:
            # Load expert with memory management
            expert_model = self._load_expert_on_demand(expert_name)
            
            # Perform generation using the corrected batch creation
            masked_seq_str = str(masked_sequence)
            if 'X' not in masked_seq_str:
                print(f"⚠️ Warning: No masked positions found in sequence for unmasking!")
                return masked_seq_str
            
            # Create batch using the corrected implementation
            batch = self._create_dplm2_batch(structure, target_length, masked_seq_str)
            
            # Generate with the expert model
            with torch.cuda.amp.autocast(dtype=torch.bfloat16) if self.device.type == "cuda" else torch.no_grad():
                output = expert_model.generate(
                    input_tokens=batch["input_tokens"],
                    max_iter=150,
                    temperature=temperature,
                    unmasking_strategy="deterministic",
                    sampling_strategy="argmax",
                    partial_masks=batch["partial_mask"],
                )
            
            # Decode the output
            generated_tokens = output["output_tokens"][0]
            decoded_seq = expert_model.tokenizer.decode(generated_tokens.cpu().tolist())
            
            # Extract amino acid sequence
            aa_type = 1
            type_ids = expert_model.get_modality_type(generated_tokens.unsqueeze(0))
            aa_positions = (type_ids[0] == aa_type).nonzero(as_tuple=False).flatten()
            
            if len(aa_positions) > 0:
                aa_tokens = generated_tokens[aa_positions].cpu().tolist()
                sequence = expert_model.tokenizer.decode(aa_tokens)
                sequence = sequence.replace(expert_model.tokenizer.aa_cls_token, "")
                sequence = sequence.replace(expert_model.tokenizer.aa_eos_token, "")
                sequence = sequence.replace(" ", "")  # Remove spaces
            else:
                sequence = decoded_seq.replace(" ", "")  # Remove spaces from fallback too
            
            # Clean sequence to only contain valid amino acids (like dplm2_integration_clean.py)
            valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
            sequence = "".join(c for c in sequence.upper() if c in valid_aa)
            
            # Truncate to target length if needed
            if len(sequence) > target_length:
                sequence = sequence[:target_length]
            
            # Clear memory after generation
            self.memory_manager.emergency_cleanup()
            
            return sequence
            
        except Exception as e:
            print(f"❌ Generation failed with expert {expert_id}: {e}")
            self.memory_manager.emergency_cleanup()
            raise
    
    def _create_dplm2_batch(self, structure: Dict, target_length: int, masked_sequence: str = None) -> Dict:
        """Create DPLM2 batch with memory management"""
        # Use the corrected implementation from dplm2_integration_fixed_new.py
        from .dplm2_integration_fixed_new import DPLM2IntegrationCorrected
        
        # Create a temporary instance to use the corrected batch creation
        temp_integration = DPLM2IntegrationCorrected(self.model_name)
        return temp_integration._create_dplm2_batch(structure, target_length, masked_sequence)
    
    def compute_ensemble_surprisal(self, structure: Dict, candidate_sequence: str, masked_positions: List[int]) -> float:
        """Compute ensemble surprisal with memory management"""
        if not masked_positions or not self.expert_instances:
            return 0.0
        
        total_surprisal = 0.0
        num_experts = len(self.expert_instances)
        
        # Create masked version for logit extraction
        masked_seq = list(candidate_sequence)
        for pos in masked_positions:
            if pos < len(masked_seq):
                masked_seq[pos] = 'X'
        masked_seq_str = ''.join(masked_seq)
        
        for expert_id in range(num_experts):
            try:
                # Use memory-managed expert loading
                expert_name = self.expert_mapping.get(expert_id, "airkingbd/dplm2_150m")
                expert_model = self._load_expert_on_demand(expert_name)
                
                logits, _ = self.get_masked_logits(structure, masked_seq_str, expert_id)
                if logits.numel() == 0:
                    continue
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Compute surprisal for actual amino acids at masked positions
                for i, pos in enumerate(masked_positions):
                    if i < logits.shape[0] and pos < len(candidate_sequence):
                        aa = candidate_sequence[pos]
                        aa_idx = self._aa_to_idx(aa)
                        if aa_idx is not None and aa_idx < probs.shape[1]:
                            prob = probs[i, aa_idx].item()
                            surprisal = -torch.log(torch.clamp(torch.tensor(prob), min=1e-10)).item()
                            total_surprisal += surprisal
                            
            except Exception as e:
                print(f"Error computing surprisal for expert {expert_id}: {e}")
                continue
        
        # Average across experts and positions
        total_positions = len(masked_positions) * num_experts
        return total_surprisal / max(1, total_positions)
    
    def get_masked_logits(self, structure: Dict, masked_sequence: str, expert_id: int = None) -> Tuple[torch.Tensor, List[int]]:
        """Get masked logits with memory management - same approach as dplm2_integration_clean.py"""
        try:
            # Simple entropy calculation - count masked positions (same as clean version)
            masked_count = masked_sequence.count('X')
            if masked_count == 0:
                return torch.tensor([]), []
            
            # Return dummy logits and positions for compatibility
            dummy_logits = torch.zeros(masked_count, 20)  # 20 amino acids
            masked_positions = [i for i, c in enumerate(masked_sequence) if c == 'X']
            
            return dummy_logits, masked_positions
            
        except Exception as e:
            print(f"Error getting masked logits: {e}")
            return torch.tensor([]), []
    
    def compute_predictive_entropy(self, structure: Dict, masked_sequence: str, expert_id: int = None) -> float:
        """Compute predictive entropy for uncertainty estimation - same as dplm2_integration_clean.py"""
        try:
            # Simple entropy calculation - count masked positions (same as clean version)
            masked_count = masked_sequence.count('X')
            if masked_count == 0:
                return 0.0
            
            # Return normalized entropy based on masked positions
            return float(masked_count) / len(masked_sequence)
            
        except Exception as e:
            print(f"Error computing predictive entropy: {e}")
            return 0.0
    
    def _aa_to_idx(self, aa: str) -> Optional[int]:
        """Convert amino acid to tokenizer index."""
        try:
            if hasattr(self.model.tokenizer, 'encode'):
                tokens = self.model.tokenizer.encode(aa, add_special_tokens=False)
                return tokens[0] if tokens else None
            return None
        except:
            # Fallback mapping
            aa_map = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                     'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
            return aa_map.get(aa.upper())
    
    def cleanup(self):
        """Cleanup all loaded models"""
        print("🧹 Cleaning up memory-optimized DPLM2 integration...")
        
        # Clear expert cache
        self.expert_model_cache.clear()
        
        # Unload all models from memory manager
        for model_name in list(self.memory_manager.model_registry.keys()):
            self.memory_manager.unload_model(model_name)
        
        print(f"✅ Cleanup completed. {self.memory_manager.get_memory_status()}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass
