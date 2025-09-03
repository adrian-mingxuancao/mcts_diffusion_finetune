import os
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

# Import the working version from the local directory
from .dplm2_integration_fixed import DPLM2Integration

class DPLM2IntegrationCorrected(DPLM2Integration):
    """
    Corrected version that uses ID path for structure (robust) and text path for AA.
    This matches the working generate_dplm2_patched_v2.py logic exactly.
    Includes memory management and logits extraction fixes.
    """
    
    def get_masked_logits(self, structure: Dict, masked_sequence: str) -> Tuple[torch.Tensor, List[int]]:
        """
        Override to ensure we use the main model for logits extraction, not unloaded experts.
        """
        # Temporarily store current model
        current_model = self.model
        
        # Always use the main model for logits extraction to avoid None references
        if hasattr(self, 'expert_instances') and self.model_name in self.expert_instances:
            main_model = self.expert_instances[self.model_name]
            if main_model is not None:
                self.model = main_model
        
        try:
            return super().get_masked_logits(structure, masked_sequence)
        finally:
            # Restore original model reference
            self.model = current_model
    
    def _create_dplm2_batch(self, structure: Dict, target_length: int, masked_sequence: str = None) -> Dict:
        """
        Create DPLM-2 batch using EXACT training format (TokenizedProteinDataset approach).
        This matches how DPLM-2 was actually trained!
        """
        tok = self.model.tokenizer

        # ---------- 1) STRUCTURE → TEXT (EXACT training format) ----------
        # Get structure tokens - handle arrays properly
        struct_seq_raw = structure.get("struct_seq")
        if struct_seq_raw is None:
            struct_seq_raw = structure.get("struct_ids")
        
        # Convert to comma-separated string if needed
        if hasattr(struct_seq_raw, 'tolist'):
            struct_seq_str = ','.join(map(str, struct_seq_raw.tolist()))
        elif isinstance(struct_seq_raw, (list, tuple)):
            struct_seq_str = ','.join(map(str, struct_seq_raw))
        elif isinstance(struct_seq_raw, str):
            struct_seq_str = struct_seq_raw
        else:
            # Fallback to FASTA loading
            pdb_id = structure.get('pdb_id', '')
            chain_id = structure.get('chain_id', '')
            structure_name = f"{pdb_id}_{chain_id}" if pdb_id and chain_id else structure.get('name', '').replace('CAMEO ', '')
            from utils.struct_loader import load_struct_seq_from_fasta
            struct_seq_str = load_struct_seq_from_fasta("/home/caom/AID3/dplm/data-bin/cameo2022/struct.fasta", structure_name)
        
        # EXACT training format from TokenizedProteinDataset lines 286-295:
        # struct_tokens = struct_seq.split(",")
        # struct_tokens = "".join(struct_tokens)  # ← KEY: concatenate digits!
        struct_tokens = struct_seq_str.split(",")
        struct_tokens_clean = "".join(struct_tokens)  # ← EXACT training format!
        
        # Debug: Check if the concatenated string is too long
        print(f"   🔍 Debug: struct_tokens count: {len(struct_tokens)}")
        print(f"   🔍 Debug: concatenated length: {len(struct_tokens_clean)}")
        print(f"   🔍 Debug: first 50 chars: {struct_tokens_clean[:50]}")
        
        # Add special tokens exactly like training
        struct_text = tok.struct_cls_token + struct_tokens_clean + tok.struct_eos_token
        
        # Calculate AA length from original structure tokens
        aa_length = len(struct_tokens)

        # ---------- 2) AA → TEXT → IDS (keep your existing AA text path) ----------
        if masked_sequence is None:
            # full inverse folding baseline: AA body all mask tokens of length aa_length
            aa_body = tok.aa_mask_token * aa_length
        else:
            ms = str(masked_sequence)
            aa_body = "".join(tok.aa_mask_token if c == "X" else c for c in ms)

        aa_text = tok.aa_cls_token + aa_body + tok.aa_eos_token

        # ---------- 3) TOKENIZE BOTH MODALITIES (same as training DPLM2Collater) ----------
        try:
            batch_struct = tok.batch_encode_plus(
                [struct_text],
                add_special_tokens=False,
                padding="longest",
                truncation=True,  # Add truncation to prevent excessive nesting
                max_length=2048,  # Reasonable limit
                return_tensors="pt"
            )
            batch_aa = tok.batch_encode_plus(
                [aa_text],
                add_special_tokens=False,
                padding="longest",
                truncation=True,  # Add truncation to prevent excessive nesting
                max_length=2048,  # Reasonable limit
                return_tensors="pt"
            )
        except Exception as e:
            print(f"   ❌ Tokenization failed: {e}")
            print(f"   🔍 struct_text length: {len(struct_text)}")
            print(f"   🔍 aa_text length: {len(aa_text)}")
            raise
        
        # ---------- 4) CONCATENATE STRUCT + AA TOKENS ----------
        input_tokens = torch.concat([batch_struct["input_ids"], batch_aa["input_ids"]], dim=1)
        input_tokens = input_tokens.to(self.device)

        # ---------- 5) GET MODALITY TYPES AND SPECIAL TOKEN MASKS ----------
        non_special = self.model.get_non_special_symbol_mask(input_tokens)
        type_ids = self.model.get_modality_type(input_tokens)
        
        # ---------- 6) APPLY INVERSE FOLDING MASK (mask AA tokens, keep structure) ----------
        aa_type = 1
        input_tokens.masked_fill_(
            (type_ids == aa_type) & non_special,
            tok._token_to_id[tok.aa_mask_token],
        )
        
        # ---------- 7) CREATE PARTIAL MASK (freeze structure, allow AA to be trainable) ----------
        struct_type = 0
        partial_mask = (type_ids == struct_type)
        
        # Debug logs
        trainable = int((~partial_mask).sum().item())
        print(f"   🧮 AA body to diffuse: {aa_length}")
        print(f"   ✅ Trainable AA positions this step: {trainable}")
        print(f"   📊 Batch info: input_tokens={input_tokens.shape}, partial_mask={partial_mask.shape}")

        return {
            "input_tokens": input_tokens,
            "partial_mask": partial_mask,
        }
    
    def _clear_gpu_memory(self):
        """Clear GPU memory to prevent OOM issues."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _load_expert_on_demand(self, expert_name):
        """Load expert model with memory management to prevent OOM."""
        # Clear memory before loading new expert
        self._clear_gpu_memory()
        
        # Check if already loaded
        if self.expert_instances.get(expert_name) is not None:
            return self.expert_instances[expert_name]
        
        try:
            print(f"🔄 Loading expert on-demand: {expert_name}")
            
            # For 3B model, be extra careful with memory
            if '3b' in expert_name.lower():
                print(f"⚠️ Loading large 3B model - clearing all GPU memory first")
                # Clear all expert instances to free memory
                for name, model in list(self.expert_instances.items()):
                    if name != expert_name:
                        del model
                        del self.expert_instances[name]
                self._clear_gpu_memory()
            
            # Load expert model using parent method
            expert_model = super()._load_expert_on_demand(expert_name)
            
            # Clear memory after loading
            self._clear_gpu_memory()
            
            return expert_model
            
        except Exception as e:
            print(f"❌ Failed to load expert {expert_name}: {e}")
            self._clear_gpu_memory()
            raise
    
    def generate_with_expert(self, expert_id: int, structure: Dict, target_length: int, masked_sequence: str = None) -> str:
        """Override to add memory management after generation."""
        try:
            # Call parent method
            result = super().generate_with_expert(expert_id, structure, target_length, masked_sequence)
            
            # Clear memory after generation to prevent accumulation
            self._clear_gpu_memory()
            
            return result
            
        except Exception as e:
            # Clear memory on error too
            self._clear_gpu_memory()
            raise
    
    def get_masked_logits(self, structure: Dict, masked_sequence: str, expert_id: int = None) -> Tuple[torch.Tensor, List[int]]:
        """
        Extract raw logits at masked positions for uncertainty tracking.
        
        Args:
            structure: Structure data with struct_seq/struct_ids
            masked_sequence: Sequence with X tokens at masked positions
            expert_id: Optional expert model to use
            
        Returns:
            (logits_at_masked, masked_positions): Logits tensor and position indices
        """
        original_model = self.model
        expert_model = None
        
        # Load expert on-demand if specified
        if expert_id is not None and hasattr(self, 'expert_instances'):
            expert_names = list(self.expert_instances.keys())
            if expert_id < len(expert_names):
                expert_name = expert_names[expert_id]
                # Ensure expert is loaded using parent class method
                if expert_name not in self.expert_instances or self.expert_instances[expert_name] is None:
                    # Call parent class method directly
                    self._load_expert_on_demand(expert_name)
                expert_model = self.expert_instances[expert_name]
                if expert_model is not None:
                    self.model = expert_model
        
        # Fallback to main model if expert not available
        if self.model is None:
            self.model = original_model
            if self.model is None:
                print("Error: No model available for logits extraction")
                return torch.empty(0, 20), []
        
        try:
            # Create batch for forward pass
            batch = self._create_dplm2_batch(structure, len(masked_sequence), masked_sequence)
            
            # Get masked positions
            masked_positions = [i for i, c in enumerate(masked_sequence) if c == 'X']
            if not masked_positions:
                vocab_size = self.model.config.vocab_size if hasattr(self.model, 'config') else 20
                return torch.empty(0, vocab_size), []
            
            # Forward pass to get logits
            with torch.no_grad():
                # Use the same interface as the working generation code
                outputs = self.model(input_ids=batch['input_tokens'])
                
                # Extract logits from model output dict
                logits = outputs['logits']  # [batch_size, seq_len, vocab_size]
                
                # Extract logits at masked positions in AA sequence part
                input_tokens = batch['input_tokens']
                type_ids = self.model.get_modality_type(input_tokens)
                struct_type = 0
                struct_len = (type_ids[0] == struct_type).sum().item()
                aa_start_idx = struct_len + 1  # +1 for AA CLS token
                
                masked_logits = []
                for pos in masked_positions:
                    aa_pos_in_batch = aa_start_idx + pos
                    if aa_pos_in_batch < logits.shape[1]:
                        masked_logits.append(logits[0, aa_pos_in_batch, :])
                
                if masked_logits:
                    return torch.stack(masked_logits), masked_positions
                else:
                    vocab_size = self.model.config.vocab_size if hasattr(self.model, 'config') else 20
                    return torch.empty(0, vocab_size), []
                    
        except Exception as e:
            print(f"Error getting masked logits: {e}")
            vocab_size = self.model.config.vocab_size if hasattr(self.model, 'config') else 20
            return torch.empty(0, vocab_size), []
        finally:
            # Always restore original model
            self.model = original_model

    def compute_ensemble_surprisal(self, structure: Dict, candidate_sequence: str, 
                                 masked_positions: List[int]) -> float:
        """
        Compute ensemble surprisal of candidate sequence at masked positions.
        
        Args:
            structure: Structure data
            candidate_sequence: Complete candidate sequence
            masked_positions: Positions that were masked during generation
            
        Returns:
            Average surprisal across experts and positions
        """
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
    
    def compute_predictive_entropy(self, structure: Dict, masked_sequence: str, expert_id: int = None) -> float:
        """
        Compute predictive entropy at masked positions for uncertainty estimation.
        
        Args:
            structure: Structure data
            masked_sequence: Sequence with X tokens at masked positions
            expert_id: Expert model to use (None for main model)
            
        Returns:
            Average entropy across masked positions
        """
        try:
            logits, masked_positions = self.get_masked_logits(structure, masked_sequence, expert_id)
            if logits.numel() == 0:
                return 0.0
            
            # Convert to probabilities and compute entropy
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Entropy = -sum(p * log(p))
            entropy = -(probs * log_probs).sum(dim=-1)
            
            # Average entropy across positions
            return entropy.mean().item()
            
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

