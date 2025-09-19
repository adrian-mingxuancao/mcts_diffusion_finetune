"""
Simplified DPLM-2 integration focusing on core generation tasks:
1. Generate from fully masked sequences/structures (baseline generation)
2. Generate from partially masked sequences/structures (MCTS expansion)

Uses official DPLM2 API with proper input_tokens and partial_masks parameters.
"""

# Patch ESM before any other imports to prevent regression weights download
def patch_esm_regression_weights():
    """Patch ESM to skip regression weight downloads that cause 403 errors"""
    try:
        import esm.pretrained as _esm_pkg
        def skip_regression_weights(model_name):
            return False
        _esm_pkg._has_regression_weights = skip_regression_weights
        print("✓ ESM regression weights patched")
    except ImportError:
        print("⚠ ESM not available for patching")

# Apply patch immediately
patch_esm_regression_weights()

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import contextlib

try:
    # Try MultimodalDiffusionProteinLanguageModel first
    try:
        from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel
        DPLM2_MODEL_CLASS = MultimodalDiffusionProteinLanguageModel
        print("✅ Using MultimodalDiffusionProteinLanguageModel")
    except ImportError as e:
        print(f"⚠️ MultimodalDiffusionProteinLanguageModel not available: {e}")
        # Fallback to DPLM2Bit which doesn't need CUDA extensions
        from byprot.models.dplm2 import DPLM2Bit
        DPLM2_MODEL_CLASS = DPLM2Bit
        print("✅ Using DPLM2Bit as fallback")
    
    # Set DPLM2 for backward compatibility
    DPLM2 = DPLM2_MODEL_CLASS
    
    from byprot.models.utils import get_struct_tokenizer
    from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer
    
    # Register DPLM2Tokenizer globally for HuggingFace model loading
    import transformers
    from transformers import AutoTokenizer
    
    # Method 1: Register with the Auto* stack so `tokenizer_class` is resolvable
    DPLM2Tokenizer.register_for_auto_class("AutoTokenizer")
    print("✅ DPLM2Tokenizer registered for auto class")
    
    # Method 2: Also register by name (optional but good for compatibility)
    AutoTokenizer.register("DPLM2Tokenizer", DPLM2Tokenizer)
    print("✅ DPLM2Tokenizer registered by name")
    
    # Also make it available in global namespace
    globals()['DPLM2Tokenizer'] = DPLM2Tokenizer
    
    DPLM2_AVAILABLE = True
    print("✅ DPLM2 modules imported and tokenizer registered")
    
except ImportError as e:
    print(f"❌ Could not import DPLM-2 modules: {e}")
    DPLM2 = None
    get_struct_tokenizer = None
    DPLM2_AVAILABLE = False


class DPLM2Integration:
    """Simplified DPLM-2 integration for core generation tasks."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.expert_models = {}
        self.expert_instances = {}  # Add missing expert_instances attribute
        self.tokenizer = None
        self.struct_tokenizer = None
        self.main_model = None
        self.model = None  # Add model attribute for compatibility
        
        # Available expert models (on-demand loading)
        self.expert_names = {
            0: "airkingbd/dplm2_650m",
            1: "airkingbd/dplm2_150m", 
            2: "airkingbd/dplm2_3b",
            3: "proteinmpnn"  # ProteinMPNN expert for inverse folding
        }
        
        # Load main model (150M)
        self._load_main_model()
        
        # Initialize structure tokenizer
        self._load_struct_tokenizer()
        
        # Report expert status
        self._report_expert_status()
    
    def _load_main_model(self):
        """Load main DPLM-2 model (150M)."""
        if DPLM2 is None:
            raise ImportError("DPLM2 not available - check byprot installation")
            
        try:
            print(f"🔄 Loading main DPLM-2 model: {self.expert_names[1]}")
            self.main_model = DPLM2.from_pretrained(self.expert_names[1]).to(self.device)
            self.main_model.eval()
            self.model = self.main_model  # Set model reference for compatibility
            self.tokenizer = self.main_model.tokenizer
            
            # Initialize expert_instances with main model
            self.expert_instances[self.expert_names[1]] = self.main_model
            
            print(f"✅ Main model loaded: {self.expert_names[1]}")
        except Exception as e:
            print(f"❌ Failed to load main model: {e}")
            raise
    
    def _load_struct_tokenizer(self):
        """Load official DPLM-2 structure tokenizer."""
        if get_struct_tokenizer is None:
            print("⚠️ Structure tokenizer not available - using fallback")
            return
            
        try:
            print("🔄 Loading DPLM-2 structure tokenizer...")
            self.struct_tokenizer = get_struct_tokenizer()
            print("✅ Structure tokenizer loaded")
        except Exception as e:
            print(f"⚠️ Failed to load structure tokenizer: {e}")
            self.struct_tokenizer = None
    
    def _report_expert_status(self):
        """Report the status of available experts."""
        print(f"🤖 EXPERT STATUS:")
        print(f"   ✅ Available experts: {len(self.expert_names)} ({list(self.expert_names.values())})")
        print(f"   🎯 Multi-expert MCTS will use {len(self.expert_names)} experts")
        print(f"   🔧 Using GitHub working version fixes for 3B model compatibility")
    
    def _patch_3b_model_forward(self, model):
        """Patch the 3B model's forward method to handle type_ids parameter during generation only."""
        try:
            # Only patch the network layer's forward method (where generation fails)
            # Keep the main forward method intact for entropy calculation
            if hasattr(model, 'net') and hasattr(model.net, 'forward'):
                import inspect
                net_forward_signature = inspect.signature(model.net.forward)
                if 'type_ids' not in net_forward_signature.parameters:
                    original_net_forward = model.net.forward
                    
                    def patched_net_forward(*args, **kwargs):
                        # Remove type_ids from kwargs if present (only affects generation path)
                        if 'type_ids' in kwargs:
                            print(f"   🔧 Removing type_ids from 3B model net forward call")
                        kwargs.pop('type_ids', None)
                        return original_net_forward(*args, **kwargs)
                    
                    model.net.forward = patched_net_forward
                    print(f"   🔧 Applied targeted 3B model network forward patch")
                else:
                    print(f"   ℹ️ 3B model net.forward already supports type_ids")
                    
        except Exception as e:
            print(f"   ⚠️ Failed to patch 3B model forward: {e}")
    
    def _load_expert_on_demand(self, expert_name: str):
        """Load expert model on-demand using the same approach as generate_dplm2_patched_v2.py"""
        if expert_name in self.expert_instances and self.expert_instances[expert_name] is not None:
            return self.expert_instances[expert_name]
        
        print(f"🔄 Loading expert on-demand: {expert_name}")
        
        try:
            # Handle ProteinMPNN expert differently
            if expert_name == "proteinmpnn":
                from .proteinmpnn_integration import create_proteinmpnn_expert
                model = create_proteinmpnn_expert(device=self.device)
                model.load_model()  # Load the ESM-2 model
                
                # Cache the loaded model
                self.expert_instances[expert_name] = model
                print(f"✅ ProteinMPNN expert loaded successfully")
                return model
            
            # Handle DPLM-2 experts
            # Use the EXACT same approach as generate_dplm2_patched_v2.py
            if expert_name.endswith("_bit"):
                from byprot.models.dplm2 import DPLM2Bit
                model = DPLM2Bit.from_pretrained(expert_name).to(self.device)
            else:
                # This is the key: use DPLM2.from_pretrained directly like the working code
                model = DPLM2.from_pretrained(expert_name).to(self.device)
            
            model.eval()
            
            # CRITICAL FIX: Patch 3B model's forward method to handle type_ids parameter
            if "3b" in expert_name:
                print(f"🔧 Applying 3B model compatibility patch for {expert_name}")
                self._patch_3b_model_forward(model)
                print(f"✅ 3B model forward method patching completed")
            
            # Cache the loaded model
            self.expert_instances[expert_name] = model
            
            print(f"✅ Expert {expert_name} loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load expert {expert_name}: {e}")
            # Store None to mark as failed and avoid repeated attempts
            self.expert_instances[expert_name] = None
            return None
    
    def _load_expert(self, expert_id: int):
        """Load expert model on demand."""
        if expert_id == 1 and self.main_model is not None:
            return self.main_model
            
        if expert_id not in self.expert_models:
            model_name = self.expert_names[expert_id]
            print(f"🔄 Loading expert {expert_id}: {model_name}")
            
            model = self._load_expert_on_demand(model_name)
            self.expert_models[expert_id] = model
                
        return self.expert_models[expert_id]
    
    @contextlib.contextmanager
    def _with_model(self, model):
        """Context manager for model usage with cleanup."""
        try:
            yield model
        finally:
            torch.cuda.empty_cache()
    
    def _tokenize_structure(self, struct_tokens: str) -> torch.Tensor:
        """
        Tokenize structure tokens using official DPLM-2 structure tokenizer.
        
        Args:
            struct_tokens: Structure tokens (comma-separated numbers or mask tokens)
            
        Returns:
            Tokenized structure tensor
        """
        try:
            if self.struct_tokenizer is not None:
                # Use official structure tokenizer
                if ',' in struct_tokens:
                    # Handle comma-separated numeric tokens
                    # Convert to coordinate format expected by structure tokenizer
                    token_list = [int(x.strip()) for x in struct_tokens.split(',') if x.strip().isdigit()]
                    # Convert to dummy coordinates (structure tokenizer expects coordinates)
                    # This is a placeholder - in practice, you'd have real coordinates
                    dummy_coords = np.random.randn(len(token_list), 3) * 10.0
                    struct_tensor = self.struct_tokenizer.tokenize(dummy_coords)
                    return struct_tensor
                else:
                    # Handle mask tokens or other formats
                    # Use regular tokenizer for mask tokens
                    return self.tokenizer.encode(struct_tokens, add_special_tokens=False, return_tensors="pt")[0]
            else:
                # Fallback: use regular tokenizer
                return self.tokenizer.encode(struct_tokens, add_special_tokens=False, return_tensors="pt")[0]
                
        except Exception as e:
            print(f"⚠️ Structure tokenization failed: {e}")
            # Fallback to mask tokens
            if ',' in struct_tokens:
                num_tokens = len(struct_tokens.split(','))
            else:
                num_tokens = len(struct_tokens.split())
            
            mask_tokens = [self.tokenizer.struct_mask_token] * num_tokens
            mask_string = ' '.join(mask_tokens)
            return self.tokenizer.encode(mask_string, add_special_tokens=False, return_tensors="pt")[0]
    
    def _create_dplm2_batch(self, structure: Dict, target_length: int, masked_sequence: str = None) -> Dict:
        """Create DPLM-2 batch for logits extraction and generation."""
        # Extract structure tokens from structure dict
        struct_tokens = structure.get('struct_seq', '')
        if isinstance(struct_tokens, (list, tuple)):
            struct_tokens = ','.join(map(str, struct_tokens))
        
        # Use masked sequence or create fully masked sequence
        if masked_sequence is None:
            aa_sequence = 'X' * target_length
        else:
            aa_sequence = masked_sequence
            
        # Use the working batch creation method
        return self._create_dplm2_batch_working(struct_tokens, aa_sequence, "inverse_folding")
    
    def _create_dplm2_batch_working(self, struct_tokens: str, aa_sequence: str, task_type: str) -> Dict:
        """
        Create DPLM-2 batch using EXACT original generate_dplm2_patched_v2.py approach.
        No truncation, proper concatenation, exact batch format.
        """
        tok = self.tokenizer
        
        # ---------- 1) STRUCTURE → TEXT (EXACT original format) ----------
        if ',' in struct_tokens:
            # EXACT original approach: "".join(str(record.seq).split(","))
            struct_tokens_clean = "".join(struct_tokens.split(","))
        else:
            # Handle mask tokens - keep as space-separated for tokenizer
            struct_tokens_clean = struct_tokens
        
        # Add special tokens exactly like original
        struct_text = tok.struct_cls_token + struct_tokens_clean + tok.struct_eos_token
        
        # ---------- 2) AA → TEXT (EXACT original format) ----------
        # Remove existing special tokens if present
        aa_body = aa_sequence.replace(tok.aa_cls_token, "").replace(tok.aa_eos_token, "")
        aa_text = tok.aa_cls_token + aa_body + tok.aa_eos_token
        
        # Debug output
        print(f"   🔍 Debug: struct_tokens count: {len(struct_tokens.split(',')) if ',' in struct_tokens else len(struct_tokens.split())}")
        print(f"   🔍 Debug: struct_text length: {len(struct_text)}")
        print(f"   🔍 Debug: aa_text length: {len(aa_text)}")
        print(f"   🔍 Debug: struct_text sample: {struct_text[:50]}...")
        
        # ---------- 3) TOKENIZE BOTH MODALITIES (NO TRUNCATION) ----------
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
            print(f"❌ Tokenization failed: {e}")
            print(f"🔍 struct_text length: {len(struct_text)}")
            print(f"🔍 aa_text length: {len(aa_text)}")
            raise
        
        # ---------- 4) CONCATENATE STRUCT + AA TOKENS (EXACT original) ----------
        input_tokens = torch.concat([batch_struct["input_ids"], batch_aa["input_ids"]], dim=1)
        input_tokens = input_tokens.to(self.device)
        
        print(f"   🔍 Debug: Final batch shape: {input_tokens.shape}")
        print(f"   🔍 Debug: Expected ~{len(struct_tokens.split(',')) + len(aa_sequence) + 4} tokens (struct + aa + special tokens)")
        
        # ---------- 5) GET MODALITY TYPES AND APPLY MASKING (EXACT original) ----------
        # Get modality types and masks with graceful None handling (GitHub working version)
        type_ids = self.main_model.get_modality_type(input_tokens)
        non_special = self.main_model.get_non_special_symbol_mask(input_tokens)
        
        # Check for None returns and handle gracefully (key fix from working version)
        if type_ids is None:
            print("⚠️ get_modality_type returned None, using fallback")
            type_ids = torch.zeros_like(input_tokens)
        if non_special is None:
            print("⚠️ get_non_special_symbol_mask returned None, using fallback")
            non_special = torch.ones_like(input_tokens, dtype=torch.bool)
        
        # Ensure all tensors are on the same device
        type_ids = type_ids.to(self.device)
        non_special = non_special.to(self.device)
        
        aa_type = 1
        struct_type = 0
        
        if task_type == "inverse_folding":
            # Mask AA tokens for inverse folding (EXACT original approach)
            input_tokens.masked_fill_(
                (type_ids == aa_type) & non_special,
                tok._token_to_id[tok.aa_mask_token],
            )
            partial_mask = (type_ids == struct_type)  # Don't mask struct tokens
        elif task_type == "folding":
            # Mask structure tokens for folding (EXACT original approach)
            input_tokens.masked_fill_(
                (type_ids == struct_type) & non_special,
                tok._token_to_id[tok.struct_mask_token],
            )
            partial_mask = (type_ids == aa_type)  # Don't mask AA tokens
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        return {
            "input_tokens": input_tokens,
            "partial_mask": partial_mask,
        }
    
    def generate_from_masked_input(self, aa_sequence: str, struct_tokens: str, 
                                 task_type: str, expert_id: int = 1, 
                                 temperature: float = 1.0, max_iter: int = 1) -> Optional[str]:
        """
        Core generation function using official DPLM2 API.
        
        Args:
            aa_sequence: Amino acid sequence (may contain mask tokens)
            struct_tokens: Structure tokens (may contain mask tokens)
            task_type: "folding" or "inverse_folding"
            expert_id: Expert model to use
            temperature: Generation temperature
            max_iter: Maximum generation iterations
            
        Returns:
            Generated sequence or structure tokens
        """
        try:
            model = self._load_expert(expert_id)
            
            # Handle ProteinMPNN expert differently
            if hasattr(model, 'get_expert_id') and model.get_expert_id() == "proteinmpnn":
                return self._generate_with_proteinmpnn(model, aa_sequence, struct_tokens, task_type, temperature)
            
            with self._with_model(model):
                # Use the EXACT working approach from the GitHub repository
                batch = self._create_dplm2_batch_working(struct_tokens, aa_sequence, task_type)
                
                input_tokens = batch["input_tokens"]
                
                # Check if model has get_modality_type method (3B compatibility)
                try:
                    type_ids = model.get_modality_type(input_tokens)
                    struct_type, aa_type = 0, 1
                except AttributeError:
                    print(f"⚠️ Model doesn't have get_modality_type, using fallback for generation")
                    # For 3B model, we'll rely on the batch partial_mask
                    type_ids = None
                
                # Set partial_mask based on task type
                partial_mask = batch["partial_mask"]
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16) if "cuda" in str(self.device) else torch.no_grad():
                    # Check if this is a 3B model that needs different generation parameters
                    try:
                        output = model.generate(
                            input_tokens=input_tokens,
                            max_iter=max_iter,
                            temperature=temperature,
                            unmasking_strategy=f"stochastic{temperature}",
                            sampling_strategy="annealing@2.2:1.0",
                            partial_masks=partial_mask,
                        )
                    except TypeError as e:
                        if "type_ids" in str(e):
                            print(f"⚠️ 3B model generation failed with type_ids error, trying alternative approach")
                            print(f"   Model type: {type(model)}")
                            print(f"   Model class: {model.__class__.__name__}")
                            
                            # Try without partial_masks parameter (3B model might not support it)
                            try:
                                print(f"   Trying generation without partial_masks...")
                                output = model.generate(
                                    input_tokens=input_tokens,
                                    max_iter=max_iter,
                                    temperature=temperature,
                                    unmasking_strategy=f"stochastic{temperature}",
                                    sampling_strategy="annealing@2.2:1.0",
                                    # Remove partial_masks parameter
                                )
                                print(f"   ✅ 3B model generation succeeded without partial_masks!")
                            except Exception as e2:
                                try:
                                    # Try even simpler parameters
                                    print(f"   Trying minimal generation parameters: {e2}")
                                    output = model.generate(
                                        input_tokens=input_tokens,
                                        max_iter=max_iter,
                                        temperature=temperature,
                                    )
                                    print(f"   ✅ 3B model generation succeeded with minimal parameters!")
                                except Exception as e3:
                                    print(f"⚠️ All 3B generation methods failed: {e3}")
                                    # Skip this expert for generation but allow entropy calculation
                                    return None
                        else:
                            raise e
                
                # Decode output using the working approach
                generated_tokens = output["output_tokens"][0]
                
                # Extract the appropriate modality based on task type (EXACT original approach)
                if task_type == "inverse_folding":
                    # EXACT original approach from generate_dplm2_patched_v2.py lines 200-211
                    aa_type = 1
                    try:
                        type_ids_generated = model.get_modality_type(generated_tokens.unsqueeze(0))
                        aa_positions = (type_ids_generated[0] == aa_type).nonzero(as_tuple=False).flatten()
                    except AttributeError:
                        # 3B model fallback: assume second half is AA tokens
                        print(f"⚠️ Model doesn't have get_modality_type, using fallback for AA extraction")
                        seq_len = len(generated_tokens)
                        aa_start = seq_len // 2
                        aa_positions = torch.arange(aa_start, seq_len)
                    
                    if len(aa_positions) > 0:
                        aa_tokens = generated_tokens[aa_positions].cpu().tolist()
                        sequence = model.tokenizer.decode(aa_tokens)
                        # Remove special tokens (EXACT original)
                        sequence = sequence.replace(model.tokenizer.aa_cls_token, "")
                        sequence = sequence.replace(model.tokenizer.aa_eos_token, "")
                        # Remove spaces and keep only valid amino acids
                        sequence = sequence.replace(" ", "")
                        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
                        sequence = "".join(c for c in sequence.upper() if c in valid_aa)
                        
                        print(f"   🔍 Debug: Extracted {len(aa_positions)} AA positions from {len(generated_tokens)} total tokens")
                        print(f"   🔍 Debug: Final sequence length: {len(sequence)}")
                        return sequence
                    else:
                        print("   ⚠️ No AA positions found in generated tokens")
                        return None
                elif task_type == "folding":
                    # Extract structure tokens (similar approach)
                    struct_type = 0
                    try:
                        type_ids_generated = model.get_modality_type(generated_tokens.unsqueeze(0))
                        struct_positions = (type_ids_generated[0] == struct_type).nonzero(as_tuple=False).flatten()
                    except AttributeError:
                        # 3B model fallback: assume first half is structure tokens
                        print(f"⚠️ Model doesn't have get_modality_type, using fallback for structure extraction")
                        seq_len = len(generated_tokens)
                        struct_end = seq_len // 2
                        struct_positions = torch.arange(0, struct_end)
                    
                    if len(struct_positions) > 0:
                        struct_tokens_list = generated_tokens[struct_positions].cpu().tolist()
                        structure = model.tokenizer.decode(struct_tokens_list)
                        structure = structure.replace(model.tokenizer.struct_cls_token, "")
                        structure = structure.replace(model.tokenizer.struct_eos_token, "")
                        structure = structure.replace(" ", "")
                        
                        print(f"   🔍 Debug: Extracted {len(struct_positions)} struct positions from {len(generated_tokens)} total tokens")
                        print(f"   🔍 Debug: Final structure length: {len(structure)}")
                        return structure
                    else:
                        print("   ⚠️ No structure positions found in generated tokens")
                        return None
                
                return None
                
        except Exception as e:
            print(f"⚠️ Generation failed: {e}")
            return None
    
    def generate_baseline_sequence(self, structure_tokens: str, target_length: int, expert_id: int = 1) -> str:
        """Generate baseline sequence from structure (inverse folding)."""
        # Create fully masked AA sequence (EXACT original approach)
        if ',' in structure_tokens:
            # For comma-separated structure tokens, use the count of tokens
            actual_length = len(structure_tokens.split(','))
        else:
            # For other formats, use the target_length
            actual_length = target_length
            
        masked_aa = self.tokenizer.aa_mask_token * actual_length
        return self.generate_from_masked_input(
            aa_sequence=masked_aa,
            struct_tokens=structure_tokens,
            task_type="inverse_folding",
            expert_id=expert_id
        )
    
    def generate_baseline_structure(self, sequence: str, expert_id: int = 1) -> str:
        """Generate baseline structure from sequence (folding)."""
        # Create fully masked structure tokens - use space-separated mask tokens
        masked_struct = ' '.join([self.tokenizer.struct_mask_token] * len(sequence))
        return self.generate_from_masked_input(
            aa_sequence=sequence,
            struct_tokens=masked_struct,
            task_type="folding",
            expert_id=expert_id
        )
    
    def generate_structure_tokens_from_sequence(self, expert_id: int, sequence: str, temperature: float = 1.0) -> Optional[str]:
        """Generate structure tokens from sequence (folding task)."""
        # Create fully masked structure tokens
        masked_struct = ' '.join([self.tokenizer.struct_mask_token] * len(sequence))
        return self.generate_from_masked_input(
            aa_sequence=sequence,
            struct_tokens=masked_struct,
            task_type="folding",
            expert_id=expert_id,
            temperature=temperature
        )
    
    def generate_with_expert(self, expert_id: int, structure: Dict[str, Any], 
                           target_length: int, masked_sequence: str = None, 
                           temperature: float = 1.0) -> Optional[str]:
        """
        Generate with specific expert for MCTS expansion.
        
        Args:
            expert_id: Expert model ID (0, 1, 2)
            structure: Structure data containing task info
            target_length: Target sequence length
            masked_sequence: Masked sequence for inverse folding
            temperature: Generation temperature
            
        Returns:
            Generated sequence or structure tokens
        """
        if 'struct_seq' in structure and masked_sequence:
            # Inverse folding: generate sequence from structure
            return self.generate_from_masked_input(
                aa_sequence=masked_sequence,
                struct_tokens=structure['struct_seq'],
                task_type="inverse_folding",
                expert_id=expert_id,
                temperature=temperature
            )
        elif 'sequence' in structure:
            # Folding: generate structure from sequence
            masked_struct = ' '.join([self.tokenizer.struct_mask_token] * len(structure['sequence']))
            return self.generate_from_masked_input(
                aa_sequence=structure['sequence'],
                struct_tokens=masked_struct,
                task_type="folding",
                expert_id=expert_id,
                temperature=temperature
            )
        else:
            print("⚠️ Unable to determine task type from structure data")
            return None
    
    def generate_structure_tokens_from_sequence(self, expert_id: int, sequence: str, temperature: float = 1.0) -> Optional[str]:
        """Generate structure tokens from sequence (folding task)."""
        # Create fully masked structure tokens
        masked_struct = ' '.join([self.tokenizer.struct_mask_token] * len(sequence))
        return self.generate_from_masked_input(
            aa_sequence=sequence,
            struct_tokens=masked_struct,
            task_type="folding",
            expert_id=expert_id,
            temperature=temperature
        )
    
    def generate_structure_tokens(self, sequence: str, masked_struct_tokens: str, expert_id: int, temperature: float = 0.9) -> Optional[str]:
        """
        Generate new structure tokens from partially masked structure tokens (for folding MCTS).
        
        Args:
            sequence: Fixed amino acid sequence
            masked_struct_tokens: Structure tokens with some positions masked
            expert_id: Expert model to use
            temperature: Sampling temperature
            
        Returns:
            New structure tokens with masked positions filled
        """
        try:
            return self.generate_from_masked_input(
                aa_sequence=sequence,
                struct_tokens=masked_struct_tokens,
                task_type="folding",
                expert_id=expert_id,
                temperature=temperature
            )
        except Exception as e:
            print(f"Error generating structure tokens: {e}")
            return None
    
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
        if not masked_positions or not hasattr(self, 'expert_instances'):
            return 0.0
        
        total_surprisal = 0.0
        num_experts = len(self.expert_instances) if self.expert_instances else 3
        
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
        if expert_id is not None:
            expert_name = self.expert_names.get(expert_id)
            if expert_name:
                expert_model = self._load_expert_on_demand(expert_name)
                if expert_model is not None:
                    self.model = expert_model
                else:
                    print(f"⚠️ Expert {expert_id} ({expert_name}) failed to load, using main model")
                    self.model = original_model
        
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
                # Try to get vocab size from different sources
                vocab_size = 20  # Default fallback
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
                    vocab_size = self.model.config.vocab_size
                elif hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'vocab_size'):
                    vocab_size = self.model.tokenizer.vocab_size
                elif hasattr(self.tokenizer, 'vocab_size'):
                    vocab_size = self.tokenizer.vocab_size
                return torch.empty(0, vocab_size), []
            
            # Forward pass to get logits
            with torch.no_grad():
                # Check if this is a 3B model that needs different parameters
                input_tokens = batch['input_tokens']
                
                # Try different forward call signatures for compatibility
                try:
                    # First try the standard DPLM2 interface
                    outputs = self.model(input_tokens=input_tokens)
                except TypeError as e:
                    if "unexpected keyword argument" in str(e) or "missing 1 required positional argument" in str(e):
                        # Model might use different parameter names
                        try:
                            outputs = self.model(input_ids=input_tokens)
                        except TypeError:
                            try:
                                # Try positional argument
                                outputs = self.model(input_tokens)
                            except TypeError:
                                # Try calling the underlying net
                                if hasattr(self.model, 'net'):
                                    outputs = self.model.net(input_tokens)
                                else:
                                    raise Exception("Could not find compatible forward method")
                    else:
                        raise e
                
                # Extract logits from model output dict
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('prediction_scores', None))
                else:
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                if logits is None:
                    print(f"⚠️ Could not extract logits from model output")
                    # Try to get vocab size from different sources
                    vocab_size = 20  # Default fallback
                    if hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
                        vocab_size = self.model.config.vocab_size
                    elif hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'vocab_size'):
                        vocab_size = self.model.tokenizer.vocab_size
                    elif hasattr(self.tokenizer, 'vocab_size'):
                        vocab_size = self.tokenizer.vocab_size
                    return torch.empty(0, vocab_size), []
                
                # Extract logits at masked positions in AA sequence part
                try:
                    type_ids = self.model.get_modality_type(input_tokens)
                    struct_type = 0
                    struct_len = (type_ids[0] == struct_type).sum().item()
                    aa_start_idx = struct_len + 1  # +1 for AA CLS token
                except AttributeError:
                    # 3B model might not have get_modality_type method
                    print(f"⚠️ Model doesn't have get_modality_type, using fallback")
                    # Fallback: assume first half is structure, second half is AA
                    seq_len = input_tokens.shape[1]
                    aa_start_idx = seq_len // 2
                
                masked_logits = []
                for pos in masked_positions:
                    aa_pos_in_batch = aa_start_idx + pos
                    if aa_pos_in_batch < logits.shape[1]:
                        masked_logits.append(logits[0, aa_pos_in_batch, :])
                
                if masked_logits:
                    return torch.stack(masked_logits), masked_positions
                else:
                    vocab_size = getattr(self.model.config, 'vocab_size', 20)
                    return torch.empty(0, vocab_size), []
                    
        except Exception as e:
            print(f"Error getting masked logits: {e}")
            vocab_size = getattr(self.model.config, 'vocab_size', 20)
            return torch.empty(0, vocab_size), []
        finally:
            # Always restore original model
            self.model = original_model
    
    def _aa_to_idx(self, aa: str) -> Optional[int]:
        """Convert amino acid to tokenizer index."""
        try:
            if hasattr(self.tokenizer, 'encode'):
                tokens = self.tokenizer.encode(aa, add_special_tokens=False)
                return tokens[0] if tokens else None
            return None
        except:
            # Fallback mapping
            aa_map = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                     'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
            return aa_map.get(aa.upper())
    
    def generate_inverse_folding_batch(self, structure_data: Dict, masked_sequences: List[str], 
                                     expert_id: int = 1, temperature: float = 1.0) -> List[str]:
        """
        Batch generation for inverse folding (legacy method for compatibility).
        
        Args:
            structure_data: Structure information containing struct_seq
            masked_sequences: List of masked amino acid sequences
            expert_id: Expert model to use
            temperature: Generation temperature
            
        Returns:
            List of generated sequences
        """
        results = []
        struct_tokens = structure_data.get('struct_seq', '')
        
        for masked_seq in masked_sequences:
            result = self.generate_from_masked_input(
                aa_sequence=masked_seq,
                struct_tokens=struct_tokens,
                task_type="inverse_folding",
                expert_id=expert_id,
                temperature=temperature
            )
            results.append(result if result else masked_seq)  # Fallback to original if failed
        
        return results
    
    def generate_folding_batch(self, sequences: List[str], expert_id: int = 1, 
                             temperature: float = 1.0) -> List[str]:
        """
        Batch generation for folding (legacy method for compatibility).
        
        Args:
            sequences: List of amino acid sequences
            expert_id: Expert model to use
            temperature: Generation temperature
            
        Returns:
            List of generated structure tokens
        """
        results = []
        
        for sequence in sequences:
            masked_struct = ' '.join([self.tokenizer.struct_mask_token] * len(sequence))
            result = self.generate_from_masked_input(
                aa_sequence=sequence,
                struct_tokens=masked_struct,
                task_type="folding",
                expert_id=expert_id,
                temperature=temperature
            )
            results.append(result if result else masked_struct)  # Fallback if failed
        
        return results

    def cleanup_all(self):
        """Clean up all models."""
        for expert_id in list(self.expert_models.keys()):
            self.cleanup_expert(expert_id)
        if self.main_model is not None:
            del self.main_model
            self.main_model = None
        torch.cuda.empty_cache()
