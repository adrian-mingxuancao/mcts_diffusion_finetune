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
    
    def __init__(self, device: str = "cuda", default_max_iter: int = 150, default_temperature: float = 1.0):
        self.device = device
        self.expert_models = {}
        self.expert_instances = {}  # Add missing expert_instances attribute
        self.tokenizer = None
        self.struct_tokenizer = None
        self.main_model = None
        self.model = None  # Add model attribute for compatibility
        
        # Default generation parameters
        self.default_max_iter = default_max_iter
        self.default_temperature = default_temperature
        
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
            self.struct_tokenizer_path = "airkingbd/struct_tokenizer"  # Official path
            self.struct_tokenizer = get_struct_tokenizer(self.struct_tokenizer_path)
            print("✅ Structure tokenizer loaded")
        except Exception as e:
            print(f"⚠️ Failed to load structure tokenizer: {e}")
            self.struct_tokenizer = None
            self.struct_tokenizer_path = "airkingbd/struct_tokenizer"  # Set path even if loading fails
    
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
            # Handle ProteinMPNN expert differently - USE REAL PROTEINMPNN
            if expert_name == "proteinmpnn":
                from .proteinmpnn_real import RealProteinMPNNExpert
                model = RealProteinMPNNExpert(device=self.device, temperature=1.0)
                model.load_model()  # Load the REAL ProteinMPNN model
                
                # Cache the loaded model
                self.expert_instances[expert_name] = model
                print(f"✅ REAL ProteinMPNN expert loaded successfully")
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
            # Handle comma-separated numeric tokens (already tokenized by struct_tokenizer)
            if ',' in struct_tokens:
                # Parse numeric tokens directly - these are already DPLM-2 structure tokens
                token_list = []
                for x in struct_tokens.split(','):
                    x = x.strip()
                    if x.isdigit():
                        token_list.append(int(x))
                    elif x == '<mask_struct>':
                        # Convert mask token to ID using regular tokenizer
                        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.struct_mask_token)
                        token_list.append(mask_id)
                    elif x in ['<cls_struct>', '<eos_struct>']:
                        # Skip special tokens - they'll be added by batch creation
                        continue
                
                if token_list:
                    return torch.tensor(token_list, dtype=torch.long)
                else:
                    # Fallback to mask tokens
                    raise ValueError("No valid tokens found in comma-separated string")
            else:
                # Handle space-separated or other formats using regular tokenizer
                return self.tokenizer.encode(struct_tokens, add_special_tokens=False, return_tensors="pt")[0]
                
        except Exception as e:
            print(f"⚠️ Structure tokenization failed: {e}, using mask tokens")
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
            
        # Get task type from structure dict, default to inverse_folding
        task_type = structure.get('task_type', structure.get('task', 'inverse_folding'))
            
        # Use the working batch creation method
        return self._create_dplm2_batch_working(struct_tokens, aa_sequence, task_type)
    
    def _create_dplm2_batch_working(self, struct_tokens: str, aa_sequence: str, task_type: str) -> Dict:
        """
        Create DPLM-2 batch using EXACT original generate_dplm2_patched_v2.py approach.
        No truncation, proper concatenation, exact batch format.
        """
        print(f"   🔍 DEBUG _create_dplm2_batch_working inputs:")
        print(f"      struct_tokens: {struct_tokens[:100]}...")
        print(f"      aa_sequence: {aa_sequence[:100]}...")
        print(f"      task_type: {task_type}")
        tok = self.tokenizer
        
        # ---------- 1) STRUCT → TEXT (EXACT official approach from generate_dplm2_patched_v2.py line 87-92) ----------
        # Remove any pre-inserted special tokens
        struct_body = struct_tokens.replace(tok.struct_cls_token, "").replace(tok.struct_eos_token, "")
        
        # Handle comma-separated or space-separated structure tokens
        if ',' in struct_body or ' ' in struct_body:
            # Split by comma or space
            if ',' in struct_body:
                tokens_list = struct_body.split(",")
            else:
                tokens_list = struct_body.split()
            
            tokens_list = [token.strip() for token in tokens_list if token.strip()]
            processed_tokens = []
            for token in tokens_list:
                if token == 'MASK_TOKEN_PLACEHOLDER' or token == '<mask_struct>':
                    processed_tokens.append(tok.struct_mask_token)
                else:
                    processed_tokens.append(token)
            # CRITICAL: Join WITHOUT separator (official line 87: "".join())
            # "3888,3856,3856" → "388838563856"
            # "<mask_struct>,3856" → "<mask_struct>3856"
            struct_tokens_clean = "".join(processed_tokens)
        else:
            struct_tokens_clean = struct_body

        # Add special tokens (official line 89-92)
        struct_text = tok.struct_cls_token + struct_tokens_clean + tok.struct_eos_token

        # ---------- 2) AA → TEXT (EXACT official approach from generate_dplm2_patched_v2.py line 76-77) ----------
        # Remove any pre-inserted special tokens
        aa_body = aa_sequence.replace(tok.aa_cls_token, "").replace(tok.aa_eos_token, "")
        # Convert 'X' placeholders to the tokenizer's AA mask token
        aa_body = aa_body.replace('X', tok.aa_mask_token)
        
        # CRITICAL: NO SPACES (official line 76-77)
        # "MQGF..." stays as "MQGF..." (tokenizer handles character-level internally)
        # "<mask_aa><mask_aa>MQG" stays as "<mask_aa><mask_aa>MQG"
        # The tokenizer's batch_encode_plus will split this character-by-character
            
        # Add special tokens (official approach line 77)
        aa_text = tok.aa_cls_token + aa_body + tok.aa_eos_token
        
        # Show actual batch being sent to DPLM-2
        print(f"   🔍 BATCH TO DPLM-2:")
        print(f"      AA: {aa_text}")
        print(f"      STRUCT: {struct_text}")
        print(f"      Task: {task_type}")
        
        # DEBUG: Show raw structure tokens before processing
        print(f"   🔍 DEBUG RAW STRUCT: {struct_tokens[:100]}...")
        
        # DEBUG: Count actual motif structure tokens in raw input
        if ',' in struct_tokens:
            raw_tokens = struct_tokens.split(',')
            raw_tokens = [t.strip() for t in raw_tokens if t.strip() and not t.startswith('<')]
            non_mask_tokens = [t for t in raw_tokens if t != tok.struct_mask_token and t != 'MASK_TOKEN_PLACEHOLDER']
            print(f"   🔍 DEBUG: {len(non_mask_tokens)} non-mask structure tokens in raw input")
        
        # ---------- 3) TOKENIZE BOTH MODALITIES (NO TRUNCATION) ----------
        # CRITICAL: The tokenizer needs PRE-TOKENIZED input (list of tokens, not string)
        # For structure: split into individual 4-digit tokens and special tokens
        # For AA: split into individual characters and special tokens
        
        # Pre-tokenize structure: split by special tokens first, then by 4-digit chunks
        struct_tokens_list = []
        struct_tokens_list.append(tok.struct_cls_token)
        
        # Remove special tokens from body
        struct_body_clean = struct_tokens_clean
        i = 0
        while i < len(struct_body_clean):
            # Check for mask token
            if struct_body_clean[i:i+len(tok.struct_mask_token)] == tok.struct_mask_token:
                struct_tokens_list.append(tok.struct_mask_token)
                i += len(tok.struct_mask_token)
            # Check for 4-digit structure token
            elif i + 4 <= len(struct_body_clean) and struct_body_clean[i:i+4].isdigit():
                struct_tokens_list.append(struct_body_clean[i:i+4])
                i += 4
            else:
                # Skip any other characters (shouldn't happen)
                i += 1
        
        struct_tokens_list.append(tok.struct_eos_token)
        
        # Pre-tokenize AA: split into individual characters
        aa_tokens_list = []
        aa_tokens_list.append(tok.aa_cls_token)
        
        # Split aa_body character by character, handling mask tokens
        i = 0
        while i < len(aa_body):
            # Check for mask token
            if aa_body[i:i+len(tok.aa_mask_token)] == tok.aa_mask_token:
                aa_tokens_list.append(tok.aa_mask_token)
                i += len(tok.aa_mask_token)
            else:
                # Single amino acid character
                aa_tokens_list.append(aa_body[i])
                i += 1
        
        aa_tokens_list.append(tok.aa_eos_token)
        
        print(f"   🔍 Pre-tokenized struct: {len(struct_tokens_list)} tokens")
        print(f"   🔍 Pre-tokenized AA: {len(aa_tokens_list)} tokens")
        print(f"   🔍 First 10 struct tokens: {struct_tokens_list[:10]}")
        print(f"   🔍 First 10 AA tokens: {aa_tokens_list[:10]}")
        
        try:
            # Pass pre-tokenized lists (space-separated) to batch_encode_plus
            batch_struct = tok.batch_encode_plus(
                [' '.join(struct_tokens_list)],
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt"
            )
            batch_aa = tok.batch_encode_plus(
                [' '.join(aa_tokens_list)],
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt"
            )
        except Exception as e:
            print(f"❌ Tokenization failed: {e}")
            print(f"🔍 struct_text length: {len(struct_text)}")
            print(f"🔍 aa_text length: {len(aa_text)}")
            print(f"🔍 struct_text: {struct_text}")
            print(f"🔍 aa_text: {aa_text}")
            
            # Try to diagnose the issue
            try:
                # Count tokens manually
                struct_parts = struct_text.split()
                aa_parts = aa_text.split()
                print(f"🔍 Struct parts: {len(struct_parts)}")
                print(f"🔍 AA parts: {len(aa_parts)}")
                
                # Check for problematic tokens
                invalid_struct_tokens = [t for t in struct_parts if len(t) > 20]  # Very long tokens
                invalid_aa_tokens = [t for t in aa_parts if len(t) > 20]
                
                if invalid_struct_tokens:
                    print(f"🔍 Invalid struct tokens: {invalid_struct_tokens[:3]}")
                if invalid_aa_tokens:
                    print(f"🔍 Invalid AA tokens: {invalid_aa_tokens[:3]}")
                    
            except Exception as debug_e:
                print(f"🔍 Debug failed: {debug_e}")
            
            raise
        
        # ---------- 4) CONCATENATE STRUCT + AA TOKENS (EXACT original) ----------
        input_tokens = torch.concat([batch_struct["input_ids"], batch_aa["input_ids"]], dim=1)
        # Clamp any out-of-range token ids to the maximum known id to avoid CUDA asserts
        try:
            tok_ids = getattr(tok, '_token_to_id', None)
            if tok_ids:
                max_id = max(tok_ids.values())
                # Some environments have an off-by-one id; clamp defensively
                if torch.any(input_tokens > max_id):
                    invalid_count = int((input_tokens > max_id).sum().item())
                    print(f"⚠️ Found {invalid_count} input token ids > max_id={max_id}. Clamping to max_id.")
                    input_tokens = input_tokens.clamp(max=max_id)
        except Exception as _e:
            print(f"⚠️ Token id clamp skipped: {_e}")
        input_tokens = input_tokens.to(self.device)
        
        print(f"   🔍 Debug: Final batch shape: {input_tokens.shape}")
        print(f"   🔍 Debug: Expected ~{len(struct_tokens.split(',')) + len(aa_sequence) + 4} tokens (struct + aa + special tokens)")
        
        # ---------- 5) GET MODALITY TYPES AND APPLY MASKING (robust + motif preservation) ----------
        # Get modality types and masks with graceful None handling (GitHub working version)
        type_ids = self.main_model.get_modality_type(input_tokens)
        non_special = self.main_model.get_non_special_symbol_mask(input_tokens)
        
        # Check for None returns and handle gracefully (key fix from working version)
        # Build robust fallback type_ids based on concatenation lengths
        struct_len = batch_struct["input_ids"].shape[1]
        aa_len = batch_aa["input_ids"].shape[1]
        fallback_type_ids = torch.cat([
            torch.zeros((1, struct_len), dtype=torch.long),
            torch.ones((1, aa_len), dtype=torch.long)
        ], dim=1)
        if type_ids is None:
            print("⚠️ get_modality_type returned None, using constructed fallback type_ids")
            type_ids = fallback_type_ids
        else:
            # If shape mismatch, override with fallback
            if tuple(type_ids.shape) != tuple(input_tokens.shape):
                print("⚠️ get_modality_type returned wrong shape, using constructed fallback type_ids")
                type_ids = fallback_type_ids
        if non_special is None:
            print("⚠️ get_non_special_symbol_mask returned None, using fallback")
            non_special = torch.ones_like(input_tokens, dtype=torch.bool)
        
        # Ensure all tensors are on the same device
        type_ids = type_ids.to(self.device)
        non_special = non_special.to(self.device)
        
        aa_type = 1
        struct_type = 0

        # Safe token IDs (avoid out-of-range)
        try:
            vocab_size = getattr(tok, 'vocab_size', None)
            if vocab_size is None and hasattr(tok, '_token_to_id'):
                vocab_size = max(tok._token_to_id.values()) + 1
        except Exception:
            vocab_size = None

        def _safe_id(token_str: str) -> int:
            try:
                tid = tok._token_to_id[token_str]
                if vocab_size is not None and tid >= vocab_size:
                    return vocab_size - 1
                return tid
            except Exception:
                # Fallback: last vocab id
                return vocab_size - 1 if vocab_size is not None else 0

        aa_mask_id = _safe_id(tok.aa_mask_token)
        struct_mask_id = _safe_id(tok.struct_mask_token)

        # Initialize keep mask to True everywhere
        partial_mask = torch.ones_like(input_tokens, dtype=torch.bool)

        if task_type == "inverse_folding":
            # Generate only at AA mask positions; keep motif AA tokens intact
            gen_positions = (type_ids == aa_type) & non_special & (input_tokens == aa_mask_id)
            partial_mask[gen_positions] = False
        elif task_type == "folding":
            # Generate only at STRUCT mask positions; keep AA tokens intact
            gen_positions = (type_ids == struct_type) & non_special & (input_tokens == struct_mask_id)
            partial_mask[gen_positions] = False
        elif task_type == "motif_scaffolding":
            # Bi-modal generation: unmask BOTH AA and STRUCT masked positions
            gen_positions_aa = (type_ids == aa_type) & non_special & (input_tokens == aa_mask_id)
            gen_positions_struct = (type_ids == struct_type) & non_special & (input_tokens == struct_mask_id)
            gen_positions = gen_positions_aa | gen_positions_struct
            partial_mask[gen_positions] = False
            
            print(f"   🔍 Motif scaffolding task: unmasking {gen_positions.sum().item()} positions")
            print(f"      AA positions: {gen_positions_aa.sum().item()}")
            print(f"      Struct positions: {gen_positions_struct.sum().item()}")
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        return {
            "input_tokens": input_tokens,
            "partial_mask": partial_mask,
        }
    
    def generate_from_masked_input(self, aa_sequence: str, struct_tokens: str, 
                                 task_type: str, expert_id: int = 1, 
                                 temperature: float = None, max_iter: int = None) -> Optional[str]:
        """
        Core generation function using official DPLM2 API.
        
        Args:
            aa_sequence: Amino acid sequence (may contain mask tokens)
            struct_tokens: Structure tokens (may contain mask tokens)
            task_type: "folding" or "inverse_folding"
            expert_id: Expert model to use
            temperature: Generation temperature (uses default if None)
            max_iter: Maximum generation iterations (uses default if None)
            
        Returns:
            Generated sequence or structure tokens
        """
        # Use default values if not provided
        if temperature is None:
            temperature = self.default_temperature
        if max_iter is None:
            max_iter = self.default_max_iter
        try:
            model = self._load_expert(expert_id)
            
            # Handle ProteinMPNN expert differently
            if hasattr(model, 'get_expert_id') and model.get_expert_id() == "proteinmpnn":
                return self._generate_with_proteinmpnn(model, aa_sequence, struct_tokens, task_type, temperature)
            
            with self._with_model(model):
                # Use the robust batch creation (motif-preserving masking)
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
                            print(f"⚠️ 3B model generation failed with type_ids error")
                            print(f"   Model type: {type(model)}")
                            print(f"   Model class: {model.__class__.__name__}")
                            
                            # **CRITICAL FIX**: For motif scaffolding, we MUST preserve partial_masks
                            # Otherwise motif positions will be overwritten
                            if task_type == "motif_scaffolding":
                                print(f"   🚨 MOTIF SCAFFOLDING: Cannot proceed without partial_masks!")
                                print(f"   🚨 This would destroy motif preservation constraints!")
                                print(f"   🔧 Skipping this expert to maintain motif integrity")
                                return None
                            
                            # For non-motif tasks, try fallback approaches
                            try:
                                print(f"   Trying generation without partial_masks (non-motif task)...")
                                output = model.generate(
                                    input_tokens=input_tokens,
                                    max_iter=max_iter,
                                    temperature=temperature,
                                    unmasking_strategy=f"stochastic{temperature}",
                                    sampling_strategy="annealing@2.2:1.0",
                                    # Remove partial_masks parameter for non-motif tasks only
                                )
                                print(f"   ✅ Non-motif generation succeeded without partial_masks!")
                            except Exception as e2:
                                try:
                                    # Try even simpler parameters (non-motif tasks only)
                                    print(f"   Trying minimal generation parameters: {e2}")
                                    output = model.generate(
                                        input_tokens=input_tokens,
                                        max_iter=max_iter,
                                        temperature=temperature,
                                    )
                                    print(f"   ✅ Minimal generation succeeded!")
                                except Exception as e3:
                                    print(f"⚠️ All generation methods failed: {e3}")
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
                        
                        # **NEW**: Also extract structure tokens for motif scaffolding
                        struct_type = 0
                        try:
                            struct_positions = (type_ids_generated[0] == struct_type).nonzero(as_tuple=False).flatten()
                        except (AttributeError, NameError):
                            # 3B model fallback: assume first half is structure tokens
                            seq_len = len(generated_tokens)
                            struct_end = seq_len // 2
                            struct_positions = torch.arange(0, struct_end)
                        
                        if len(struct_positions) > 0:
                            struct_tokens = generated_tokens[struct_positions].cpu().tolist()
                            struct_sequence = model.tokenizer.decode(struct_tokens)
                            # Clean structure tokens
                            struct_sequence = struct_sequence.replace(model.tokenizer.struct_cls_token, "")
                            struct_sequence = struct_sequence.replace(model.tokenizer.struct_eos_token, "")
                            struct_sequence = struct_sequence.replace(" ", "")
                            
                            # Store in generation data for later extraction
                            self._last_generation_data = {
                                'aa_sequence': sequence,
                                'structure_sequence': struct_sequence,
                                'aa_positions': aa_positions.cpu().tolist(),
                                'struct_positions': struct_positions.cpu().tolist(),
                                'raw_tokens': generated_tokens.cpu().tolist()
                            }
                            
                            print(f"   🔍 Debug: Also extracted {len(struct_positions)} structure positions")
                            print(f"   🔍 Debug: Structure sequence length: {len(struct_sequence)}")
                        
                        return sequence
                    else:
                        print("   ⚠️ No AA positions found in generated tokens")
                        return None
                elif task_type == "motif_scaffolding":
                    # Extract BOTH AA and structure sequences
                    print(f"   🔍 Extracting both AA and structure from motif scaffolding generation...")
                    
                    # Extract AA tokens
                    aa_type = 1
                    try:
                        type_ids_generated = model.get_modality_type(generated_tokens.unsqueeze(0))
                        aa_positions = (type_ids_generated[0] == aa_type).nonzero(as_tuple=False).flatten()
                    except AttributeError:
                        # 3B model fallback: assume second half is AA tokens
                        seq_len = len(generated_tokens)
                        aa_start = seq_len // 2
                        aa_positions = torch.arange(aa_start, seq_len)
                    
                    # Extract structure tokens
                    struct_type = 0
                    try:
                        struct_positions = (type_ids_generated[0] == struct_type).nonzero(as_tuple=False).flatten()
                    except (AttributeError, NameError):
                        # 3B model fallback: assume first half is structure tokens
                        seq_len = len(generated_tokens)
                        struct_end = seq_len // 2
                        struct_positions = torch.arange(0, struct_end)
                    
                    if len(aa_positions) > 0 and len(struct_positions) > 0:
                        # Decode AA sequence
                        aa_tokens = generated_tokens[aa_positions].cpu().tolist()
                        aa_sequence = model.tokenizer.decode(aa_tokens)
                        aa_sequence = aa_sequence.replace(model.tokenizer.aa_cls_token, "")
                        aa_sequence = aa_sequence.replace(model.tokenizer.aa_eos_token, "")
                        aa_sequence = aa_sequence.replace(" ", "")
                        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
                        aa_sequence = "".join(c for c in aa_sequence.upper() if c in valid_aa)
                        
                        # Decode structure sequence
                        struct_tokens_raw = generated_tokens[struct_positions].cpu().tolist()
                        struct_sequence = model.tokenizer.decode(struct_tokens_raw)
                        struct_sequence = struct_sequence.replace(model.tokenizer.struct_cls_token, "")
                        struct_sequence = struct_sequence.replace(model.tokenizer.struct_eos_token, "")
                        struct_sequence = struct_sequence.replace(" ", "")
                        
                        # Store both sequences
                        self._last_generation_data = {
                            'aa_sequence': aa_sequence,
                            'structure_sequence': struct_sequence,
                            'aa_positions': aa_positions.cpu().tolist(),
                            'struct_positions': struct_positions.cpu().tolist(),
                            'raw_tokens': generated_tokens.cpu().tolist()
                        }
                        
                        print(f"   🔍 Debug: Extracted {len(aa_positions)} AA positions, {len(struct_positions)} struct positions")
                        print(f"   🔍 Debug: AA sequence length: {len(aa_sequence)}")
                        print(f"   🔍 Debug: Structure sequence length: {len(struct_sequence)}")
                        
                        # Return AA sequence (main result) - structure is stored in _last_generation_data
                        return aa_sequence
                    else:
                        print("   ⚠️ No AA or structure positions found in generated tokens")
                        return None
                elif task_type == "folding":
                    # For folding: extract structure tokens but return ORIGINAL AA sequence
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
                        struct_decoded = model.tokenizer.decode(struct_tokens_list)
                        struct_decoded = struct_decoded.replace(model.tokenizer.struct_cls_token, "")
                        struct_decoded = struct_decoded.replace(model.tokenizer.struct_eos_token, "")

                        struct_tokens_clean = []
                        for token in struct_decoded.split():
                            if token.isdigit() or token == model.tokenizer.struct_mask_token:
                                struct_tokens_clean.append(token)
                        
                        # CRITICAL FIX: Pad or truncate to match expected sequence length
                        expected_length = len(aa_sequence.replace(model.tokenizer.aa_cls_token, "").replace(model.tokenizer.aa_eos_token, ""))
                        if len(struct_tokens_clean) < expected_length:
                            # Pad with mask tokens
                            padding_needed = expected_length - len(struct_tokens_clean)
                            struct_tokens_clean.extend([model.tokenizer.struct_mask_token] * padding_needed)
                            print(f"   🔧 Padded structure tokens: {len(struct_tokens_clean) - padding_needed} → {len(struct_tokens_clean)}")
                        elif len(struct_tokens_clean) > expected_length:
                            # Truncate
                            struct_tokens_clean = struct_tokens_clean[:expected_length]
                            print(f"   🔧 Truncated structure tokens: {len(struct_tokens_clean) + (len(struct_tokens_clean) - expected_length)} → {len(struct_tokens_clean)}")
                        
                        structure = ",".join(struct_tokens_clean)
                        
                        print(f"   🔍 Debug: Extracted {len(struct_positions)} struct positions from {len(generated_tokens)} total tokens")
                        print(f"   🔍 Debug: Final structure token count: {len(struct_tokens_clean)} (expected: {expected_length})")
                        
                        # Store structure tokens for later use but return ORIGINAL AA sequence
                        self._last_generation_data = {
                            'structure_sequence': structure,
                            'struct_positions': struct_positions.cpu().tolist(),
                            'raw_tokens': generated_tokens.cpu().tolist()
                        }
                        
                        # CRITICAL FIX: Return original AA sequence (not structure tokens)
                        original_aa = aa_sequence.replace(self.tokenizer.aa_cls_token, "").replace(self.tokenizer.aa_eos_token, "")
                        print(f"   ✅ FOLDING: Returning original AA sequence (len={len(original_aa)}), structure tokens stored")
                        return original_aa
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
        """
        Generate baseline structure from sequence (folding).
        
        Returns structure tokens (comma-separated).
        """
        # Create fully masked structure tokens - use space-separated mask tokens
        masked_struct = ' '.join([self.tokenizer.struct_mask_token] * len(sequence))
        
        # Generate (this returns AA sequence but stores structure tokens)
        result = self.generate_from_masked_input(
            aa_sequence=sequence,
            struct_tokens=masked_struct,
            task_type="folding",
            expert_id=expert_id
        )
        
        # Extract structure tokens from last generation data
        if hasattr(self, '_last_generation_data') and self._last_generation_data:
            structure_tokens = self._last_generation_data.get('structure_sequence', '')
            if structure_tokens:
                return structure_tokens
        
        # Fallback: return empty if no structure tokens generated
        print("⚠️ No structure tokens found in generation data")
        return ""
    
    def _structure_tokens_to_coords(self, structure_tokens: str, expected_length: int) -> tuple:
        """
        Convert structure tokens to CA coordinates and pLDDT scores using structure tokenizer.
        
        Args:
            structure_tokens: Comma-separated structure token string
            expected_length: Expected sequence length
            
        Returns:
            Tuple of (ca_coords, plddt_scores) or (None, None) if conversion fails
            - ca_coords: numpy array (seq_len, 3)
            - plddt_scores: numpy array (seq_len,) with confidence scores 0-100
        """
        try:
            import torch
            from byprot.models.utils import get_struct_tokenizer
            
            # Parse structure tokens
            if ',' in structure_tokens:
                token_list = [int(t.strip()) for t in structure_tokens.split(',') if t.strip().isdigit()]
            else:
                token_list = [int(t.strip()) for t in structure_tokens.split() if t.strip().isdigit()]
            
            if len(token_list) != expected_length:
                print(f"⚠️ Token count mismatch: {len(token_list)} vs expected {expected_length}")
                return None, None
            
            # Load structure tokenizer
            if not hasattr(self, '_struct_tokenizer_detok'):
                self._struct_tokenizer_detok = get_struct_tokenizer(self.struct_tokenizer_path).to(self.device)
            
            # Convert tokens to tensor
            tokens_tensor = torch.tensor(token_list, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # Detokenize to get coordinates and pLDDT
            with torch.no_grad():
                output = self._struct_tokenizer_detok.detokenize(tokens_tensor)
            
            # Debug: print available keys
            print(f"  🔍 Detokenizer output keys: {list(output.keys())}")
            
            # Extract CA coordinates from atom37 positions
            ca_coords = None
            plddt_scores = None
            
            if 'atom37_positions' in output:
                atom37_positions = output['atom37_positions']  # (batch, seq_len, 37, 3)
                ca_coords = atom37_positions[0, :, 1, :].cpu().numpy()  # CA is atom index 1
            else:
                print("⚠️ No atom37_positions in detokenizer output")
                return None, None
            
            # Extract pLDDT scores (confidence per residue)
            if 'plddt' in output:
                plddt = output['plddt']  # (batch, seq_len) or (batch, seq_len, atoms)
                if plddt.dim() == 3:
                    # Average across atoms for per-residue confidence
                    plddt_scores = plddt[0].mean(dim=-1).cpu().numpy()
                else:
                    plddt_scores = plddt[0].cpu().numpy()
                
                # Ensure 0-100 scale
                if plddt_scores.max() <= 1.0:
                    plddt_scores = plddt_scores * 100.0
                    
                print(f"  ✅ Extracted pLDDT scores: mean={plddt_scores.mean():.1f}, range={plddt_scores.min():.1f}-{plddt_scores.max():.1f}")
            else:
                print("  ⚠️ No pLDDT in detokenizer output, using default 70.0")
                plddt_scores = np.full(expected_length, 70.0)
            
            return ca_coords, plddt_scores
                
        except Exception as e:
            print(f"⚠️ Structure token to coordinate conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def generate_sequence(self, structure_data: Dict[str, Any], target_length: int, temperature: float = None) -> Optional[str]:
        """
        Generate sequence from structure data (for MCTS interface compatibility).
        This is a wrapper around generate_baseline_sequence for inverse folding tasks.
        """
        try:
            # Extract structure tokens from structure_data
            struct_tokens = structure_data.get('struct_seq', '')
            if not struct_tokens:
                print("⚠️ No structure tokens found in structure_data")
                return None
            
            # Use the existing baseline generation method
            return self.generate_baseline_sequence(
                structure_tokens=struct_tokens,
                target_length=target_length,
                expert_id=1  # Use main DPLM-2 model for initial sequence
            )
        except Exception as e:
            print(f"❌ generate_sequence failed: {e}")
            return None
    
    def generate_structure_tokens_from_sequence(self, expert_id: int, sequence: str, temperature: float = None) -> Optional[str]:
        """Generate structure tokens from sequence (folding task)."""
        # Use default temperature if not provided
        if temperature is None:
            temperature = self.default_temperature
            
        # Create fully masked structure tokens
        masked_struct = ' '.join([self.tokenizer.struct_mask_token] * len(sequence))
        return self.generate_from_masked_input(
            aa_sequence=sequence,
            struct_tokens=masked_struct,
            task_type="folding",
            expert_id=expert_id,
            temperature=temperature
        )

    def generate_motif_scaffold(
        self,
        reference_sequence: str,
        reference_struct_tokens: Optional[str],
        motif_sequence: str,
        motif_positions: Optional[List[int]] = None,
        expert_id: int = 1,
        temperature: float = None,
        max_iter: int = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate motif scaffold preserving motif residues and unmasking scaffold for both AA and STRUCT.

        - AA template: motif fixed at motif_positions; 'X' elsewhere
        - STRUCT template: numeric tokens at motif_positions (from reference_struct_tokens when provided);
                           struct_mask_token elsewhere
        """
        try:
            if temperature is None:
                temperature = self.default_temperature
            if max_iter is None:
                max_iter = self.default_max_iter

            target_length = len(reference_sequence)

            # Determine motif positions if not provided
            if not motif_positions:
                start = reference_sequence.find(motif_sequence) if motif_sequence else -1
                if start != -1:
                    motif_positions = list(range(start, start + len(motif_sequence)))
                else:
                    center = (target_length - len(motif_sequence)) // 2
                    motif_positions = list(range(center, center + len(motif_sequence)))

            # Build AA template
            aa_chars = ['X'] * target_length
            for i, pos in enumerate(motif_positions):
                if 0 <= pos < target_length and i < len(motif_sequence):
                    aa_chars[pos] = motif_sequence[i]
            aa_template = ''.join(aa_chars)

            # Build STRUCT template
            struct_list: List[str] = [self.tokenizer.struct_mask_token] * target_length
            if reference_struct_tokens:
                toks = [t.strip() for t in reference_struct_tokens.split(',') if t.strip()]
                if len(toks) == target_length:
                    for pos in motif_positions:
                        if 0 <= pos < target_length and toks[pos].isdigit():
                            struct_list[pos] = toks[pos]
                elif len(toks) == len(motif_positions):
                    for i, pos in enumerate(motif_positions):
                        if 0 <= pos < target_length and toks[i].isdigit():
                            struct_list[pos] = toks[i]
            struct_template = ','.join(struct_list)

            # Create bi-modal unmasking batch
            batch = self._create_dplm2_batch_working(struct_template, aa_template, "motif_scaffolding")

            model = self._load_expert(expert_id)
            with self._with_model(model):
                input_tokens = batch["input_tokens"]
                partial_mask = batch["partial_mask"]

                with torch.cuda.amp.autocast(dtype=torch.bfloat16) if "cuda" in str(self.device) else torch.no_grad():
                    try:
                        outputs = model.generate(
                            input_tokens=input_tokens,
                            max_iter=max_iter,
                            temperature=temperature,
                            unmasking_strategy=f"stochastic{temperature}",
                            sampling_strategy="annealing@2.2:1.0",
                            partial_masks=partial_mask,
                        )
                    except TypeError:
                        outputs = model.generate(input_tokens=input_tokens, max_iter=max_iter, temperature=temperature)

                output_tokens = outputs["output_tokens"][0]

                # Extract modalities
                try:
                    type_ids_generated = model.get_modality_type(output_tokens.unsqueeze(0))
                    aa_type, struct_type = 1, 0
                    aa_positions = (type_ids_generated[0] == aa_type).nonzero(as_tuple=False).flatten()
                    struct_positions = (type_ids_generated[0] == struct_type).nonzero(as_tuple=False).flatten()
                except Exception:
                    seq_len = len(output_tokens)
                    mid = seq_len // 2
                    struct_positions = torch.arange(0, mid)
                    aa_positions = torch.arange(mid, seq_len)

                aa_tokens = output_tokens[aa_positions].cpu().tolist()
                aa_seq = model.tokenizer.decode(aa_tokens)
                aa_seq = aa_seq.replace(model.tokenizer.aa_cls_token, "").replace(model.tokenizer.aa_eos_token, "").replace(" ", "")
                valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
                aa_seq = "".join(c for c in aa_seq.upper() if c in valid_aa)

                struct_tok_list = output_tokens[struct_positions].cpu().tolist()
                # Decode then join whitespace-separated tokens with commas (canonical format)
                struct_decoded = model.tokenizer.decode(struct_tok_list, skip_special_tokens=True)
                # Filter out any AA tokens that might have leaked in
                struct_tokens_clean = []
                for token in struct_decoded.split():
                    # Only keep numeric tokens and structure-specific tokens
                    if token.isdigit() or token.startswith('<') or token in [model.tokenizer.struct_mask_token]:
                        struct_tokens_clean.append(token)
                    else:
                        print(f"   ⚠️ Filtered out non-structure token: '{token}'")
                struct_seq = ",".join(struct_tokens_clean)

                return {
                    "aa_sequence": aa_seq,
                    "struct_sequence": struct_seq,
                    "motif_preserved": motif_sequence in aa_seq if motif_sequence else True,
                    "output_tokens": output_tokens,
                }

        except Exception as e:
            print(f"⚠️ Motif scaffold generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_with_expert(self, expert_id: int, structure: Dict[str, Any], 
                           target_length: int, masked_sequence: str = None, 
                           temperature: float = None) -> Optional[str]:
        """
        Generate with specific expert for MCTS expansion.
        
        Args:
            expert_id: Expert model ID (0, 1, 2, 3)
            structure: Structure data containing task info
            target_length: Target sequence length
            masked_sequence: Masked sequence for inverse folding
            temperature: Generation temperature (uses default if None)
            
        Returns:
            Generated sequence or structure tokens
        """
        # Use default temperature if not provided
        if temperature is None:
            temperature = self.default_temperature
        
        # Determine task type from structure data - prioritize explicit task_type
        task_type = structure.get('task_type', structure.get('task', 'inverse_folding'))
        print(f"      🔍 Task type detected: {task_type}")
        
        if task_type == "folding" or task_type == "forward_folding":
            # Forward folding: generate structure from sequence (sequence is fixed, structure is masked)
            sequence = structure.get('sequence', masked_sequence)
            struct_seq = structure.get('struct_seq', '')
            
            # Use provided masked structure tokens or create fully masked
            if struct_seq and '<mask_struct>' in struct_seq:
                masked_struct = struct_seq  # Use partially masked structure tokens
            else:
                masked_struct = ' '.join([self.tokenizer.struct_mask_token] * len(sequence))
            
            print(f"      🔍 Forward folding: seq_len={len(sequence)}, struct_tokens={len(masked_struct.split())}")
            return self.generate_from_masked_input(
                aa_sequence=sequence,
                struct_tokens=masked_struct,
                task_type="folding",
                expert_id=expert_id,
                temperature=temperature
            )
        elif 'struct_seq' in structure and masked_sequence and task_type != "folding":
            # Inverse folding: generate sequence from structure
            print(f"      🔍 Inverse folding: masked_seq_len={len(masked_sequence)}")
            return self.generate_from_masked_input(
                aa_sequence=masked_sequence,
                struct_tokens=structure['struct_seq'],
                task_type="inverse_folding",
                expert_id=expert_id,
                temperature=temperature
            )
        elif 'sequence' in structure:
            # Fallback: assume folding if we have a sequence
            masked_struct = ' '.join([self.tokenizer.struct_mask_token] * len(structure['sequence']))
            print(f"      🔍 Fallback folding: seq_len={len(structure['sequence'])}")
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
    
    def generate_structure_tokens(self, sequence: str, masked_struct_tokens: str, expert_id: int, temperature: float = None) -> Optional[str]:
        """
        Generate new structure tokens from partially masked structure tokens (for folding MCTS).
        
        Args:
            sequence: Fixed amino acid sequence
            masked_struct_tokens: Structure tokens with some positions masked
            expert_id: Expert model to use
            temperature: Sampling temperature (uses default if None)
            
        Returns:
            New structure tokens with masked positions filled
        """
        # Use default temperature if not provided
        if temperature is None:
            temperature = self.default_temperature
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
    
    def compute_ensemble_surprisal_structure(self, sequence: str, structure_tokens: str, mask_positions: List[int]) -> float:
        """
        Compute ensemble surprisal for structure tokens at masked positions.
        
        Args:
            sequence: Amino acid sequence (fixed)
            structure_tokens: Generated structure tokens
            mask_positions: Positions that were masked during generation
            
        Returns:
            Ensemble surprisal across masked structure positions
        """
        try:
            # For structure generation, compute surprisal over structure token predictions
            # This is a simplified version - in practice, we'd need to get structure logits from all experts
            
            if not mask_positions:
                return 0.0
                
            # For now, return a reasonable surprisal estimate based on masking ratio
            # Higher masking ratio = higher surprisal
            total_positions = len(sequence)
            masking_ratio = len(mask_positions) / total_positions
            
            # Scale surprisal by masking ratio (more masking = more surprisal)
            base_surprisal = 1.5  # Base surprisal for structure prediction
            scaled_surprisal = base_surprisal * (1.0 + masking_ratio)
            
            return min(scaled_surprisal, 3.0)  # Cap at reasonable maximum
            
        except Exception as e:
            print(f"Error computing ensemble surprisal for structure: {e}")
            return 1.5  # Default surprisal
    
    def compute_predictive_entropy_structure(self, sequence: str, masked_structure_tokens: str, expert_id: int = None) -> float:
        """
        Compute predictive entropy for structure generation at masked positions.
        
        Args:
            sequence: Amino acid sequence (fixed)
            masked_structure_tokens: Structure tokens with <mask_struct> at masked positions
            expert_id: Expert model to use (None for main model)
            
        Returns:
            Average entropy across masked structure positions
        """
        try:
            # For structure generation, we compute entropy over structure token predictions
            # This is a simplified version - in practice, we'd need to get structure logits
            
            # Count masked positions
            masked_count = masked_structure_tokens.count('<mask_struct>')
            if masked_count == 0:
                return 0.0
            
            # For now, return a reasonable entropy estimate based on masking ratio
            # Higher masking ratio = higher uncertainty
            total_positions = len(sequence)
            masking_ratio = masked_count / total_positions
            
            # Scale entropy by masking ratio (more masking = more uncertainty)
            base_entropy = 1.0  # Base uncertainty for structure prediction
            scaled_entropy = base_entropy * (1.0 + masking_ratio)
            
            return min(scaled_entropy, 2.0)  # Cap at reasonable maximum
            
        except Exception as e:
            print(f"Error computing structure predictive entropy: {e}")
            return 1.0  # Default uncertainty
    
    def compute_proteinmpnn_entropy(self, sequence: str, masked_positions: List[int]) -> float:
        """
        Compute ProteinMPNN entropy at masked positions using real model logits.
        
        Args:
            sequence: Complete sequence 
            masked_positions: Positions to compute entropy for
            
        Returns:
            Average entropy at masked positions
        """
        try:
            # Load ProteinMPNN expert
            proteinmpnn_expert = self._load_expert_on_demand("proteinmpnn")
            if not proteinmpnn_expert:
                print("   ⚠️ ProteinMPNN expert not available for entropy calculation")
                return 1.5  # Default entropy
            
            # Create masked sequence for entropy calculation
            masked_seq = list(sequence)
            for pos in masked_positions:
                if pos < len(masked_seq):
                    masked_seq[pos] = 'X'
            masked_sequence = ''.join(masked_seq)
            
            # Compute entropy using ProteinMPNN model
            entropy = proteinmpnn_expert.compute_entropy(
                masked_sequence=masked_sequence,
                structure_coords=self._cameo_coordinates
            )
            
            print(f"   📊 ProteinMPNN entropy: {entropy:.3f}")
            return entropy
            
        except Exception as e:
            print(f"   ⚠️ ProteinMPNN entropy calculation failed: {e}")
            return 1.5  # Default entropy

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
                                     expert_id: int = 1, temperature: float = None) -> List[str]:
        """
        Batch generation for inverse folding (legacy method for compatibility).
        
        Args:
            structure_data: Structure information containing struct_seq
            masked_sequences: List of masked amino acid sequences
            expert_id: Expert model to use
            temperature: Generation temperature (uses default if None)
            
        Returns:
            List of generated sequences
        """
        # Use default temperature if not provided
        if temperature is None:
            temperature = self.default_temperature
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
                             temperature: float = None) -> List[str]:
        """
        Batch generation for folding (legacy method for compatibility).
        
        Args:
            sequences: List of amino acid sequences
            expert_id: Expert model to use
            temperature: Generation temperature (uses default if None)
            
        Returns:
            List of generated structure tokens
        """
        # Use default temperature if not provided
        if temperature is None:
            temperature = self.default_temperature
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
    
    def _generate_with_proteinmpnn(self, proteinmpnn_model, aa_sequence: str, 
                                  struct_tokens: str, task_type: str, temperature: float) -> str:
        """
        Generate sequence using ProteinMPNN expert.
        
        Args:
            proteinmpnn_model: Loaded ProteinMPNN expert model
            aa_sequence: Masked amino acid sequence (contains X tokens)
            struct_tokens: Structure tokens (not used for ProteinMPNN)
            task_type: Should be "inverse_folding" for ProteinMPNN
            temperature: Generation temperature
            
        Returns:
            Generated sequence (complete, no X tokens)
        """
        try:
            print(f"🎯 Generating with ProteinMPNN expert")
            print(f"   Task: {task_type}, Temperature: {temperature}")
            print(f"   Input sequence: {aa_sequence[:50]}... (length: {len(aa_sequence)})")
            
            # **CRITICAL**: Get real coordinates for ProteinMPNN structure conditioning
            structure_coords = self._get_structure_coordinates()
            
            # If no coordinates found, try to get from current baseline structure
            if structure_coords is None and hasattr(self, '_baseline_structure') and self._baseline_structure:
                if 'coordinates' in self._baseline_structure:
                    structure_coords = self._baseline_structure['coordinates']
                    print(f"   📁 Using coordinates from _baseline_structure: {structure_coords.shape}")
                elif 'backbone_coords' in self._baseline_structure:
                    structure_coords = self._baseline_structure['backbone_coords']
                    print(f"   📁 Using backbone_coords from _baseline_structure: {structure_coords.shape}")
            
            if structure_coords is not None:
                print(f"   🧬 Using real structure coordinates: {structure_coords.shape}")
            else:
                print(f"   ⚠️ No structure coordinates available for ProteinMPNN")
            
            # ProteinMPNN generates complete sequences from masked input with structure conditioning
            generated_sequences = proteinmpnn_model.generate_sequences(
                masked_sequence=aa_sequence,
                structure_coords=structure_coords,
                num_samples=1
            )
            
            if generated_sequences and len(generated_sequences) > 0:
                result = generated_sequences[0]
                print(f"   Generated: {result[:50]}... (length: {len(result)})")
                return result
            else:
                print(f"   ❌ ProteinMPNN returned no sequences")
                return aa_sequence.replace('X', 'A')  # Simple fallback
                
        except Exception as e:
            print(f"   ❌ ProteinMPNN generation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: replace X with random amino acids
            import random
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            result = ""
            for char in aa_sequence:
                if char == 'X':
                    result += random.choice(amino_acids)
                else:
                    result += char
            return result
    
    def _get_structure_coordinates(self) -> Optional[np.ndarray]:
        """
        Get structure coordinates for ProteinMPNN from baseline structure.
        Returns CA coordinates from .pkl file data.
        """
        try:
            # Check if we have cached coordinates
            if hasattr(self, 'cached_structure_coords') and self.cached_structure_coords is not None:
                return self.cached_structure_coords
            
            # Try to get coordinates from baseline structure (passed from ablation script)
            print(f"   🔍 Checking baseline structure: hasattr={hasattr(self, 'baseline_structure')}")
            if hasattr(self, 'baseline_structure'):
                print(f"   🔍 Baseline structure exists: {self.baseline_structure is not None}")
                if self.baseline_structure:
                    print(f"   🔍 Baseline structure keys: {list(self.baseline_structure.keys())}")
            
            if hasattr(self, 'baseline_structure') and self.baseline_structure:
                structure = self.baseline_structure
                
                # Check for different coordinate formats in .pkl data
                coords = None
                
                # First check for 'coordinates' key (used by MCTS script)
                if 'coordinates' in structure and structure['coordinates'] is not None:
                    coords = structure['coordinates']
                    print(f"   📁 Found coordinates: {coords.shape if hasattr(coords, 'shape') else type(coords)}")
                    # Fix coordinate shape for ProteinMPNN: extract CA atoms if needed
                    if hasattr(coords, 'shape') and len(coords.shape) == 3 and coords.shape[1] == 3:
                        coords = coords[:, 1, :]  # Extract CA atoms (index 1: N, CA, C)
                        print(f"   📁 Extracted CA atoms: {coords.shape}")
                elif 'backbone_coords' in structure and structure['backbone_coords'] is not None:
                    backbone_coords = structure['backbone_coords']
                    # Handle both numpy arrays and lists
                    if hasattr(backbone_coords, 'shape'):
                        if len(backbone_coords.shape) == 3 and backbone_coords.shape[1] >= 2:
                            coords = backbone_coords[:, 1, :]  # CA atoms at index 1
                        else:
                            coords = backbone_coords
                    else:
                        # Handle list format - assume it's already CA coordinates
                        coords = backbone_coords
                    print(f"   📁 Using backbone_coords from .pkl: {getattr(coords, 'shape', len(coords))}")
                    
                elif 'atom_positions' in structure and structure['atom_positions'] is not None:
                    atom_positions = structure['atom_positions']
                    # Handle both numpy arrays and lists
                    if hasattr(atom_positions, 'shape'):
                        if len(atom_positions.shape) == 3 and atom_positions.shape[1] >= 2:
                            coords = atom_positions[:, 1, :]  # CA atoms at index 1
                        else:
                            coords = atom_positions
                    else:
                        # Handle list format - assume it's already CA coordinates
                        coords = atom_positions
                    print(f"   📁 Using atom_positions from .pkl: {getattr(coords, 'shape', len(coords))}")
                
                if coords is not None:
                    # Cache for future use
                    self.cached_structure_coords = coords
                    return coords
                else:
                    print(f"   ⚠️ No coordinates found in baseline structure keys: {list(structure.keys())}")
            
            print(f"   ⚠️ No baseline structure available for coordinate extraction")
            return None
            
        except Exception as e:
            print(f"   ❌ Error extracting structure coordinates: {e}")
            return None
    
    
    def set_baseline_structure(self, baseline_structure: Dict):
        """Set baseline structure for coordinate extraction."""
        self.baseline_structure = baseline_structure
        # Clear cached coordinates when baseline changes
        if hasattr(self, 'cached_structure_coords'):
            self.cached_structure_coords = None
        print(f"🔄 Set baseline structure with keys: {list(baseline_structure.keys())}")
    
    def set_baseline_sequence(self, sequence: str):
        """Set baseline sequence for MCTS to use as starting point."""
        self._current_baseline_sequence = sequence
        print(f"✅ Set baseline sequence: {len(sequence)} residues")
    
    def cleanup_expert(self, expert_id: int):
        """Clean up specific expert model to free GPU memory."""
        try:
            # Remove from expert_models cache
            if expert_id in self.expert_models:
                model = self.expert_models[expert_id]
                if model is not None:
                    del model
                del self.expert_models[expert_id]
                print(f"🧹 Cleaned up expert {expert_id} from cache")
            
            # Remove from expert_instances cache  
            model_name = self.expert_names.get(expert_id)
            if model_name and model_name in self.expert_instances:
                model = self.expert_instances[model_name]
                if model is not None:
                    del model
                del self.expert_instances[model_name]
                print(f"🧹 Cleaned up expert instance {model_name}")
            
            # Force garbage collection and CUDA cleanup
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"⚠️ Error cleaning up expert {expert_id}: {e}")

    def cleanup_all(self):
        """Clean up all models."""
        for expert_id in list(self.expert_models.keys()):
            self.cleanup_expert(expert_id)
        if self.main_model is not None:
            del self.main_model
            self.main_model = None
        torch.cuda.empty_cache()
