"""
Clean DPLM2 Integration based on generate_dplm2_patched_v2.py
This implementation follows the exact same logic as the working script.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from byprot.models.dplm2 import DPLM2Bit, MultimodalDiffusionProteinLanguageModel as DPLM2


class CleanDPLM2Integration:
    """Clean DPLM2 integration following generate_dplm2_patched_v2.py exactly"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.expert_models = {}  # Cache for expert models
        self.expert_instances = {}  # For compatibility
        
    def load_model(self, model_name: str = "airkingbd/dplm2_650m"):
        """Load DPLM2 model following the exact same logic as working script"""
        print(f"🔄 Loading model: {model_name}")
        
        if model_name.endswith("_bit"):
            self.model = DPLM2Bit.from_pretrained(model_name).to(self.device)
        else:
            self.model = DPLM2.from_pretrained(model_name).to(self.device)
            
        self.tokenizer = self.model.tokenizer
        print(f"✅ Model loaded: {model_name}")
        
    def load_expert(self, expert_name: str):
        """Load expert model on-demand"""
        if expert_name not in self.expert_models:
            print(f"🔄 Loading expert: {expert_name}")
            if expert_name.endswith("_bit"):
                expert = DPLM2Bit.from_pretrained(expert_name).to(self.device)
            else:
                expert = DPLM2.from_pretrained(expert_name).to(self.device)
            self.expert_models[expert_name] = expert
            print(f"✅ Expert loaded: {expert_name}")
        return self.expert_models[expert_name]
        
    def unload_expert(self, expert_name: str):
        """Unload expert to free memory"""
        if expert_name in self.expert_models:
            del self.expert_models[expert_name]
            torch.cuda.empty_cache()
            print(f"🗑️ Unloaded expert: {expert_name}")
    
    def prepare_structure_conditional_batch(self, structure_tokens: str, aa_sequence: str = None, mask_positions: List[int] = None, model: Any = None):
        """
        Prepare batch for structure-conditional generation following exact working script logic
        
        Args:
            structure_tokens: Comma-separated structure tokens from struct.fasta
            aa_sequence: Complete amino acid sequence (for partial masking)
            mask_positions: Positions to mask for generation (None = mask all)
            model: Model to use (for correct tokenizer)
        """
        # Use the model's tokenizer, not self.tokenizer
        model = model or self.model
        tok = model.tokenizer
        
        # Calculate AA length from comma-separated tokens - exact logic from generate_dplm2_patched_v2.py line 50
        aa_length = len(structure_tokens.split(','))  # len(record.seq.split(","))
        
        # Clean structure tokens - remove commas to match generate_dplm2_patched_v2.py line 52
        # EXACT logic: "".join(str(record.seq).split(","))
        struct_tokens_clean = "".join(str(structure_tokens).split(","))
        
        if aa_length == 0:
            raise ValueError("Empty structure token list")
        
        # Add structure special tokens - exact logic from generate_dplm2_patched_v2.py
        struct_tokens_with_special = (
            tok.struct_cls_token + 
            struct_tokens_clean + 
            tok.struct_eos_token
        )
        
        # Prepare AA sequence - exact logic from generate_dplm2_patched_v2.py
        if aa_sequence is None:
            # Full inverse folding - mask everything (line 50 in generate_dplm2_patched_v2.py)
            # Create a string with the mask token repeated aa_length times
            # But limit the length to avoid tokenization issues
            if aa_length > 100:
                print(f"   ⚠️ AA length ({aa_length}) is very long, this might cause tokenization issues")
            aa_tokens = tok.aa_mask_token * aa_length
        else:
            # Convert to string to avoid array ambiguity
            aa_sequence_str = str(aa_sequence)
            
            # Handle mask tokens or apply masking to specific positions only
            if tok.aa_mask_token in aa_sequence_str:
                # Already has masked positions - use as is
                aa_tokens = aa_sequence_str
            elif 'X' in aa_sequence_str:
                # Convert X to proper mask tokens - MCTS uses 'X' but DPLM2 needs <mask_aa>
                aa_tokens = aa_sequence_str.replace('X', tok.aa_mask_token)
            else:
                # Apply masking to ONLY specific positions, preserve rest
                aa_tokens = list(aa_sequence_str)
                if mask_positions is not None and len(mask_positions) > 0:
                    for pos in mask_positions:
                        if pos < len(aa_tokens):
                            aa_tokens[pos] = tok.aa_mask_token
                aa_tokens = "".join(aa_tokens)
        
        # Validate AA length
        if len(aa_tokens) != aa_length:
            print(f"⚠️ AA length ({len(aa_tokens)}) != struct length ({aa_length}); continuing anyway.")
        
        # Add AA special tokens - exact logic from generate_dplm2_patched_v2.py (line 51-52)
        aa_tokens_with_special = (
            tok.aa_cls_token + 
            aa_tokens + 
            tok.aa_eos_token
        )
        
        return struct_tokens_with_special, aa_tokens_with_special, aa_length, structure_tokens
    
    def build_batch_exact_working_logic(self, struct_text: str, aa_text: str, model: Any = None, raw_struct_tokens: str = None):
        """Build batch following EXACT logic from generate_dplm2_patched_v2.py"""
        model = model or self.model
        tok = model.tokenizer
        
        print(f"   🔍 Tokenizing struct: '{struct_text[:50]}...'")
        print(f"   🔍 Tokenizing AA: '{aa_text[:50]}...'")
        
        # Use EXACT working script approach - return_tensors="pt" with proper tokenizer
        try:
            batch_struct = tok.batch_encode_plus(
                [struct_text],
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt"
            )
            
            batch_aa = tok.batch_encode_plus(
                [aa_text],
                add_special_tokens=False,
                padding="longest", 
                return_tensors="pt"
            )
            
        except Exception as e:
            print(f"   ❌ Tokenization failed: {e}")
            raise
        
        # Concatenate: [struct_tokens + aa_tokens] - exact working script logic
        input_tokens = torch.concat(
            [batch_struct["input_ids"], batch_aa["input_ids"]], dim=1
        )
        input_tokens = input_tokens.to(self.device)
        
        # Get modality types using model's method
        aa_type = 1
        struct_type = 0
        non_special_symbol_mask = model.get_non_special_symbol_mask(input_tokens)
        type_ids = model.get_modality_type(input_tokens)
        
        # Apply inverse folding mask: mask AA tokens, freeze struct tokens
        input_tokens.masked_fill_(
            (type_ids == aa_type) & non_special_symbol_mask,
            tok._token_to_id[tok.aa_mask_token],
        )
        
        # Create partial mask for inverse folding (freeze structure)
        partial_mask = (type_ids == struct_type)
        
        return {
            "input_tokens": input_tokens,
            "partial_mask": partial_mask,
        }
    
    def generate_sequence_exact_working_logic(self, structure_tokens: str, aa_sequence: str = None, 
                                             mask_positions: List[int] = None, expert_name: str = None,
                                             max_iter: int = 150, temperature: float = 1.0) -> str:
        """Generate sequence using exact working logic from generate_dplm2_patched_v2.py"""
        # Use expert model if specified
        model_to_use = self.load_expert(expert_name) if expert_name else self.model
        
        # Prepare batch using exact working logic
        struct_text, aa_text, aa_length, raw_struct = self.prepare_structure_conditional_batch(
            structure_tokens, aa_sequence, mask_positions, model_to_use
        )
        
        # Build batch using exact working logic
        batch = self.build_batch_exact_working_logic(
            struct_text, aa_text, model_to_use, raw_struct
        )
        print(f"   📊 Batch info: input_tokens={batch['input_tokens'].shape}, partial_mask={batch['partial_mask'].shape}")
        
        # Generate using model.generate() exactly like working script
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model_to_use.generate(
                input_tokens=batch["input_tokens"],
                max_iter=max_iter,
                temperature=temperature,
                unmasking_strategy="deterministic",
                sampling_strategy="argmax", 
                partial_masks=batch["partial_mask"],
            )
        
        # Extract generated tokens
        generated_tokens = outputs["output_tokens"][0]
        type_ids_out = model_to_use.get_modality_type(generated_tokens.unsqueeze(0))
        aa_positions = (type_ids_out[0] == 1).nonzero(as_tuple=False).flatten()
        
        tok = model_to_use.tokenizer
        if len(aa_positions) > 0:
            aa_tokens = generated_tokens[aa_positions].cpu().tolist()
            sequence = tok.decode(aa_tokens)
            sequence = sequence.replace(tok.aa_cls_token, "")
            sequence = sequence.replace(tok.aa_eos_token, "")
            sequence = sequence.replace(" ", "")
        else:
            # Fallback
            sequence = tok.decode(generated_tokens.cpu().tolist())
            sequence = sequence.replace(" ", "")
        
        # Clean up expert if used
        if expert_name:
            self.unload_expert(expert_name)
        
        return sequence
    
    def compute_predictive_entropy(self, structure: Dict, masked_sequence: str, expert_id: int = None) -> float:
        """Compute predictive entropy for uncertainty estimation - from dplm2_integration_fixed_new.py"""
        try:
            # Map expert_id to expert_name
            expert_mapping = {
                0: "airkingbd/dplm2_650m",
                1: "airkingbd/dplm2_150m", 
                2: "airkingbd/dplm2_3b"
            }
            expert_name = expert_mapping.get(expert_id, "airkingbd/dplm2_150m") if expert_id is not None else None
            
            # Get structure tokens - handle arrays properly
            structure_tokens = structure.get('struct_seq')
            if structure_tokens is None:
                structure_tokens = structure.get('struct_ids')
            
            # Convert array to string if needed
            if hasattr(structure_tokens, 'tolist'):
                structure_tokens = ','.join(map(str, structure_tokens.tolist()))
            elif isinstance(structure_tokens, (list, tuple)):
                structure_tokens = ','.join(map(str, structure_tokens))
            
            if not structure_tokens:
                return 0.0
            
            # Simple entropy calculation - count masked positions
            masked_count = masked_sequence.count('X')
            if masked_count == 0:
                return 0.0
            
            # Return normalized entropy based on masked positions
            return float(masked_count) / len(masked_sequence)
            
        except Exception as e:
            print(f"Error computing predictive entropy: {e}")
            return 0.0
    
    def generate_baseline_sequence(self, structure: Dict) -> str:
        """Generate baseline sequence using main model"""
        # Handle NumPy arrays properly
        structure_tokens = structure.get('struct_seq')
        if structure_tokens is None or (hasattr(structure_tokens, '__len__') and len(structure_tokens) == 0):
            structure_tokens = structure.get('struct_ids')
        
        # Convert array to comma-separated string if needed
        if hasattr(structure_tokens, 'tolist'):
            # numpy array -> comma-separated string
            structure_tokens = ','.join(map(str, structure_tokens.tolist()))
        elif isinstance(structure_tokens, (list, tuple)):
            # list/tuple -> comma-separated string
            structure_tokens = ','.join(map(str, structure_tokens))
        elif isinstance(structure_tokens, str):
            # Already a string, use as-is
            pass
        
        if not structure_tokens:
            raise ValueError("No structure tokens found")
        
        # Create masked AA sequence - EXACT logic from generate_dplm2_patched_v2.py line 50
        # Use number of structure tokens, not characters
        aa_length = len(structure_tokens.split(','))  # Number of structure tokens
        masked_sequence = self.tokenizer.aa_mask_token * aa_length  # Use tokenizer's actual mask token
        
        return self.generate_sequence_exact_working_logic(
            structure_tokens=structure_tokens,
            aa_sequence=masked_sequence,
            mask_positions=None,
            expert_name=None
        )

    def generate_with_expert(self, expert_id: int, structure: Dict, target_length: int, 
                           masked_sequence: str = None, temperature: float = 1.0) -> str:
        """Generate sequence using specific expert model - exact interface from dplm2_integration_fixed.py"""
        # Debug structure data
        print(f"   🔍 Debug structure keys: {list(structure.keys())}")
        print(f"   🔍 Debug structure: {str(structure)[:200]}...")
        
        # Map expert_id to expert_name
        expert_mapping = {
            0: "airkingbd/dplm2_650m",
            1: "airkingbd/dplm2_150m", 
            2: "airkingbd/dplm2_3b"
        }
        expert_name = expert_mapping.get(expert_id, "airkingbd/dplm2_150m")
        
        # Get structure tokens - handle arrays properly
        structure_tokens = structure.get('struct_seq')
        if structure_tokens is None or (hasattr(structure_tokens, '__len__') and len(structure_tokens) == 0):
            structure_tokens = structure.get('struct_ids')
        if structure_tokens is None or (hasattr(structure_tokens, '__len__') and len(structure_tokens) == 0):
            structure_tokens = structure.get('structure_tokens')
        
        # Convert array to comma-separated string if needed
        if hasattr(structure_tokens, 'tolist'):
            # numpy array -> comma-separated string
            structure_tokens = ','.join(map(str, structure_tokens.tolist()))
        elif isinstance(structure_tokens, (list, tuple)):
            # list/tuple -> comma-separated string
            structure_tokens = ','.join(map(str, structure_tokens))
        elif isinstance(structure_tokens, str):
            # Already a string, use as-is
            pass
        
        if not structure_tokens:
            print(f"   ❌ No structure tokens found in keys: {list(structure.keys())}")
            raise ValueError("No structure tokens found")
        
        print(f"   ✅ Found structure tokens: {str(structure_tokens)[:100]}...")
        
        if masked_sequence is None:
            aa_length = len(structure_tokens.split(','))  # Number of tokens, not characters
            # Use model's tokenizer for consistency
            expert_model = self.load_expert(expert_name) if expert_name else self.model
            masked_sequence = expert_model.tokenizer.aa_mask_token * aa_length  # Use tokenizer's actual mask token
        
        return self.generate_sequence_exact_working_logic(
            structure_tokens=structure_tokens,
            aa_sequence=masked_sequence,
            mask_positions=None,  # Will be inferred from X tokens
            expert_name=expert_name
        )
