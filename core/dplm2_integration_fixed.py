"""
Fixed DPLM-2 integration with proper structure-conditional generation.
This implements the surgical fixes to make structure-conditioned inverse folding work.
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import random

# Add the DPLM-2 source code to the path
sys.path.insert(0, '/home/caom/AID3/dplm/src')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Set cache directories to /net/scratch/ to avoid disk quota issues
os.environ['HF_HOME'] = '/net/scratch/caom/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/net/scratch/caom/.cache/huggingface/transformers'
os.environ['HF_DATASETS_CACHE'] = '/net/scratch/caom/.cache/huggingface/datasets'
os.environ['TORCH_HOME'] = '/net/scratch/caom/.cache/torch'

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

# Mock missing CUDA extension to allow DPLM-2 imports
def patch_cuda_extension():
    """Mock the missing CUDA extension to allow DPLM-2 imports."""
    try:
        import sys
        
        class MockAttnCore:
            def __init__(self):
                pass
            def __getattr__(self, name):
                def mock_func(*args, **kwargs):
                    raise NotImplementedError(f'CUDA extension {name} not available')
                return mock_func
        
        sys.modules['attn_core_inplace_cuda'] = MockAttnCore()
        print("✓ CUDA extension mocked")
    except Exception as e:
        print(f"⚠ Failed to mock CUDA extension: {e}")

# Apply CUDA patch
patch_cuda_extension()

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
    
    from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer
    
    # Register DPLM2Tokenizer globally for HuggingFace model loading
    import transformers
    from transformers import AutoTokenizer
    
    # Register the tokenizer class
    AutoTokenizer.register("DPLM2Tokenizer", DPLM2Tokenizer)
    
    # Also make it available in global namespace
    globals()['DPLM2Tokenizer'] = DPLM2Tokenizer
    
    DPLM2_AVAILABLE = True
    print("✅ DPLM2 modules imported and tokenizer registered")
except ImportError as e:
    print(f"❌ Could not import DPLM-2 modules: {e}")
    DPLM2_AVAILABLE = False

class DPLM2Integration:
    """Fixed DPLM-2 integration that properly uses structure tokenizer."""
    
    def __init__(self, model_name: str = "airkingbd/dplm2_650m", use_local: bool = False):
        """Initialize DPLM-2 integration with multiple experts."""
        self.model_name = model_name
        self.use_local = use_local
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.struct_tokenizer = None
        
        # Multiple experts setup
        self.use_multiple_experts = True
        self.expert_models = [
            "airkingbd/dplm2_650m",    # Expert 1: 650M parameters (default)
            "airkingbd/dplm2_150m",    # Expert 2: 150M parameters  
            "airkingbd/dplm2_3b"       # Expert 3: 3B parameters
        ]
        self.expert_instances = {}
        self.num_rollouts_per_expert = 3
        
        if DPLM2_AVAILABLE:
            self._load_model()
            self._load_structure_tokenizer()
        else:
            raise ValueError("DPLM-2 not available - check environment setup")
    
    def _load_model(self):
        """Load the primary DPLM-2 model and all expert models."""
        try:
            print(f"Loading DPLM-2 model: {self.model_name}")
            
            # Load primary DPLM-2 model
            self.model = DPLM2_MODEL_CLASS.from_pretrained(
                self.model_name,
                from_huggingface=True
            )
            self.model.eval()
            self.model.to(self.device)
            
            # Get tokenizer
            self.tokenizer = self.model.tokenizer
            print(f"✅ Loaded DPLM-2 model: {self.model_name}")
            print(f"✅ Loaded tokenizer: {type(self.tokenizer).__name__}")
            print(f"✅ Has aa_mask_token: {hasattr(self.tokenizer, 'aa_mask_token')}")
            
            # Verify AA tokens match joint vocab - NO FALLBACKS
            if not hasattr(self.tokenizer, 'aa_mask_token'):
                raise ValueError("Tokenizer missing aa_mask_token - check DPLM-2 tokenizer setup")
            if not hasattr(self.tokenizer, 'aa_cls_token'):
                raise ValueError("Tokenizer missing aa_cls_token - check DPLM-2 tokenizer setup")  
            if not hasattr(self.tokenizer, 'aa_eos_token'):
                raise ValueError("Tokenizer missing aa_eos_token - check DPLM-2 tokenizer setup")
            
            print(f"✅ AA tokens verified: mask='{self.tokenizer.aa_mask_token}', cls='{self.tokenizer.aa_cls_token}', eos='{self.tokenizer.aa_eos_token}'")
            
            # Load expert models for multiple experts
            if self.use_multiple_experts:
                self._load_expert_models()
                
        except Exception as e:
            print(f"❌ Failed to load DPLM-2 model: {e}")
            raise
    
    def _load_expert_models(self):
        """Load multiple expert DPLM-2 models with different sizes."""
        print(f"🎯 Loading {len(self.expert_models)} expert models...")
        
        for i, expert_name in enumerate(self.expert_models):
            try:
                if expert_name == self.model_name:
                    # Use main model if same
                    print(f"   Expert {i+1}: {expert_name} (same as main model)")
                    self.expert_instances[expert_name] = self.model
                    continue
                
                print(f"   Expert {i+1}: Loading {expert_name}...")
                
                # Load expert model
                expert_model = DPLM2_MODEL_CLASS.from_pretrained(
                    expert_name,
                    from_huggingface=True
                )
                expert_model.eval()
                expert_model.to(self.device)
                
                # Share the main tokenizer to avoid import issues
                expert_model.tokenizer = self.tokenizer
                
                # Apply 3B model patch if needed
                if '3b' in expert_name.lower():
                    self._patch_3b_model_forward(expert_model)
                
                self.expert_instances[expert_name] = expert_model
                print(f"   Expert {i+1}: ✅ Loaded {expert_name}")
                
            except Exception as e:
                print(f"   Expert {i+1}: ❌ Failed to load {expert_name}: {e}")
                # Use main model as fallback
                self.expert_instances[expert_name] = self.model
                continue
        
        print(f"🎯 Expert models loaded: {len(self.expert_instances)}/{len(self.expert_models)}")
        if self.expert_instances:
            print(f"   Available experts: {list(self.expert_instances.keys())}")
    
    def _load_structure_tokenizer(self):
        print("Loading struct_tokenizer (no-ESM path)...")
        st = None
        for obj in (self.model, getattr(self.model, "net", None)):
            if obj is None: 
                continue
            for attr in ("struct_tokenizer", "_struct_tokenizer"):
                if hasattr(obj, attr) and getattr(obj, attr) is not None:
                    st = getattr(obj, attr)
                    print(f"✅ struct_tokenizer found on {obj.__class__.__name__}.{attr}")
                    break
            if st is not None:
                break
        if st is None:
            # Some builds expose a builder that DOESN'T hit ESM
            for meth in ("get_struct_tokenizer", "build_struct_tokenizer", "build_tokenizers"):
                if hasattr(self.model, meth):
                    try:
                        cand = getattr(self.model, meth)()
                        if isinstance(cand, tuple):
                            cand = next((c for c in cand if "token" in c.__class__.__name__.lower()), None)
                        if cand is not None:
                            st = cand
                            print(f"✅ struct_tokenizer built via model.{meth}()")
                            break
                    except Exception as e:
                        print(f"⚠ model.{meth}() failed: {e}")
        if st is None:
            raise ValueError("❌ No real DPLM-2 struct tokenizer on this checkpoint. Use a ckpt that bundles it.")
        self.struct_tokenizer = st
    
    def _coordinates_to_structure_tokens(self, structure: Dict) -> Tuple[torch.Tensor, int]:
        """
        Convert structure coordinates to structure tokens using the struct_tokenizer.
        
        Args:
            structure: Dict containing structure data with keys like 'atom_positions', 'aatype', etc.
            
        Returns:
            Tuple of (struct_ids, Ls) where:
            - struct_ids: [1, Ls] tensor of structure token IDs
            - Ls: Length of structure sequence
        """
        device = next(self.model.parameters()).device
        
        # ---- FAST PATH: Use pre-tokenized structure data if available ----
        if structure.get("struct_ids") is not None or structure.get("struct_seq") is not None:
            return self._tokens_to_struct_ids(structure)
        
        # ---- FALLBACK: Use coordinates to generate structure tokens ----
        print("   ⚠️ No pre-tokenized struct data found, falling back to coordinate-based tokenization")
        print("   ⚠️ This may not work properly with current struct_tokenizer setup")
        
        # Extract atom_positions (atom37 format)
        atom37 = None
        if "atom_positions" in structure and structure["atom_positions"] is not None:
            atom37 = structure["atom_positions"]
        elif "coordinates" in structure and structure["coordinates"] is not None:
            # Handle backbone-only coordinates [L, 3, 3] -> need to expand to [L, 37, 3]
            coords = structure["coordinates"]
            if isinstance(coords, np.ndarray) and coords.shape[-2:] == (3, 3):
                # This is backbone coords [L, 3, 3] (N, CA, C)
                # We need to create a mock atom37 array
                L = coords.shape[0]
                atom37 = np.zeros((L, 37, 3), dtype=np.float32)
                # Map backbone atoms to their positions in atom37 format
                # N=0, CA=1, C=2 in atom37 indexing
                atom37[:, 0, :] = coords[:, 0, :]  # N
                atom37[:, 1, :] = coords[:, 1, :]  # CA  
                atom37[:, 2, :] = coords[:, 2, :]  # C
                print(f"   🏗️ Converted backbone coords {coords.shape} to atom37 {atom37.shape}")
            else:
                atom37 = coords
        elif "backbone_coords" in structure and structure["backbone_coords"] is not None:
            atom37 = structure["backbone_coords"]
        else:
            raise ValueError(f"No structure tokens or coordinates found. Keys present: {list(structure.keys())}")

        # For now, raise an error to indicate this path needs pre-tokenized data
        raise ValueError(
            "Coordinate-based structure tokenization not properly configured. "
            "Please ensure struct_seq or struct_ids are available in the structure data. "
            "For CAMEO data, the struct.fasta should contain pre-tokenized structure sequences."
        )

        # Normalize shapes: [B=1, L, 13, 3] and [B=1, L, 13]
        if isinstance(coords13, np.ndarray):
            coords13 = torch.tensor(coords13, dtype=torch.float32, device=device)
        else:
            coords13 = coords13.to(device, dtype=torch.float32)

        if isinstance(mask13, np.ndarray):
            mask13 = torch.tensor(mask13, dtype=torch.bool, device=device)
        else:
            mask13 = mask13.to(device, dtype=torch.bool)

        if coords13.ndim == 3:  # [L, 13, 3]
            coords13 = coords13.unsqueeze(0)
        if mask13.ndim == 2:    # [L, 13]
            mask13 = mask13.unsqueeze(0)

        # Sanity
        assert coords13.shape[:3] == (1, L, 13) and coords13.shape[-1] == 3, f"coords13 bad shape {tuple(coords13.shape)}"
        assert mask13.shape == (1, L, 13), f"mask13 bad shape {tuple(mask13.shape)}"

        # ---- 4) Quantize & decode to a structural sequence string ----
        out = self.struct_tokenizer.quantize_and_decode(coords13, mask=mask13)
        if isinstance(out, dict):
            struct_seq = out.get("struct_seq")
        elif isinstance(out, (list, tuple)):
            struct_seq = out[0]
        else:
            struct_seq = out
        if not isinstance(struct_seq, str):
            raise ValueError("quantize_and_decode did not return a struct string")

        # ---- 5) struct string → ids in the joint vocab ----
        if not hasattr(self.struct_tokenizer, "struct_seq_to_ids"):
            raise ValueError("struct_tokenizer missing struct_seq_to_ids")
        ids = self.struct_tokenizer.struct_seq_to_ids(struct_seq)
        ids = torch.as_tensor(ids, dtype=torch.long, device=device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)  # [1, Ls]

        # Optional BOS/EOS, if defined
        bos_id = getattr(self.struct_tokenizer, "bos_token_id", getattr(self.struct_tokenizer, "struct_cls_id", None))
        eos_id = getattr(self.struct_tokenizer, "eos_token_id", getattr(self.struct_tokenizer, "struct_eos_id", None))
        if bos_id is not None and bos_id >= 0:
            ids = torch.cat([torch.tensor([[bos_id]], device=device), ids], dim=1)
        if eos_id is not None and eos_id >= 0:
            ids = torch.cat([ids, torch.tensor([[eos_id]], device=device)], dim=1)

        # Final safety: all ids must be in-range of the shared embedding
        vocab_n = self.model.net.esm.embeddings.word_embeddings.num_embeddings
        tmin, tmax = int(ids.min()), int(ids.max())
        if tmin < 0 or tmax >= vocab_n:
            raise ValueError(f"Struct ids out of range (min={tmin}, max={tmax}, vocab={vocab_n})")

        return ids, ids.shape[1]
    
    def _get_struct_boundary_ids(self, st):
        """Robust struct boundary token detection with proper fallbacks."""
        # prefer explicit struct_* fields if present
        bos_id = getattr(st, "struct_cls_id", None)
        eos_id = getattr(st, "struct_eos_id", None)

        # fallback to generic names if needed
        if bos_id is None:
            bos_id = getattr(st, "bos_token_id", None)
        if eos_id is None:
            eos_id = getattr(st, "eos_token_id", None)

        # normalize invalid to None
        if isinstance(bos_id, int) and bos_id < 0: 
            bos_id = None
        if isinstance(eos_id, int) and eos_id < 0: 
            eos_id = None
            
        # If struct tokenizer has no boundary tokens, use main tokenizer struct tokens
        if bos_id is None and hasattr(self.tokenizer, 'struct_cls_token'):
            try:
                bos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.struct_cls_token)
                if bos_id < 0:
                    bos_id = None
            except:
                bos_id = None
                
        if eos_id is None and hasattr(self.tokenizer, 'struct_eos_token'):
            try:
                eos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.struct_eos_token)
                if eos_id < 0:
                    eos_id = None
            except:
                eos_id = None
                
        return bos_id, eos_id

    def _tokens_to_struct_ids(self, structure: Dict) -> Tuple[torch.Tensor, int]:
        """Convert struct_seq or struct_ids to tensor format with boundary tokens."""
        device = next(self.model.parameters()).device
        vocab_n = self.model.net.esm.embeddings.word_embeddings.num_embeddings
        
        # Prefer struct_seq → ids over raw struct_ids for compatibility
        if structure.get('struct_seq') is not None:
            seq = structure['struct_seq']
            if not isinstance(seq, str) or not len(seq):
                raise ValueError("struct_seq must be a non-empty string")
            ids = list(self.struct_tokenizer.struct_seq_to_ids(seq))
        elif structure.get('struct_ids') is not None:
            ids = list(structure['struct_ids'])
        else:
            raise ValueError("No struct_ids or struct_seq found in structure")

        # Get boundary tokens with robust fallback logic
        bos_id, eos_id = self._get_struct_boundary_ids(self.struct_tokenizer)
        
        original_len = len(ids)
        
        if bos_id is not None:
            ids = [int(bos_id)] + [int(x) for x in ids]
        else:
            ids = [int(x) for x in ids]
        if eos_id is not None:
            ids = ids + [int(eos_id)]

        # Validate range
        ids = np.asarray(ids, dtype=np.int64)
        imin, imax = int(ids.min()), int(ids.max())
        if imin < 0 or imax >= vocab_n:
            raise ValueError(
                f"struct_ids out of range: [{imin}, {imax}] vs vocab {vocab_n}. "
                "Make sure your FASTA contains the correct token IDs for this checkpoint."
            )

        t = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        print(f"   🧪 struct_ids L(before)={original_len}, added_bos={bos_id is not None}, added_eos={eos_id is not None}, L(final)={t.shape[1]}")
        return t, t.shape[1]
    
    def generate_with_expert(self, expert_id: int, structure: Dict, target_length: int, 
                           masked_sequence: str = None, temperature: float = 1.0) -> str:
        """Generate sequence using specific expert model for UNMASKING (not conditional generation)."""
        if not self.expert_instances:
            # Fallback to main model
            return self.fill_masked_positions(structure, masked_sequence, target_length, temperature)
        
        expert_names = list(self.expert_instances.keys())
        if expert_id >= len(expert_names):
            expert_id = expert_id % len(expert_names)
        
        expert_name = expert_names[expert_id]
        expert_model = self.expert_instances[expert_name]
        
        # Verify this is UNMASKING, not conditional generation
        # Convert to string to avoid array ambiguity
        if masked_sequence is None:
            print(f"⚠️ Warning: No masked sequence provided for unmasking!")
            return ""
        
        masked_seq_str = str(masked_sequence)
        if 'X' not in masked_seq_str:
            print(f"⚠️ Warning: No masked positions found in sequence for unmasking!")
            return masked_seq_str
        
        masked_count = masked_seq_str.count('X')
        print(f"   🎯 Unmasking {masked_count} positions with {expert_name}")
        
        # Use the specific expert model for generation
        original_model = self.model
        self.model = expert_model
        
        try:
            result = self.fill_masked_positions(structure, masked_sequence, target_length, temperature)
            
            # Verify unmasking worked correctly
            if result and 'X' in result:
                print(f"   ⚠️ Warning: Unmasking incomplete, {result.count('X')} X's remain")
            elif result:
                print(f"   ✅ Unmasking complete: {masked_count} positions filled")
            
            return result
        finally:
            self.model = original_model
    
    def _create_dplm2_batch(self, structure: Dict, target_length: int, masked_sequence: str = None) -> Dict:
        """
        Create DPLM-2 batch for generation following generate_dplm2_patched_v2.py exactly.
        Uses string→tokenizer path for structure tokens (not int IDs).
        """
        tok = self.model.tokenizer
        
        # --- Force Python strings to avoid NumPy truthiness bugs ---
        struct_seq_raw = structure.get("struct_seq")
        if struct_seq_raw is None and structure.get("struct_ids") is None:
            raise ValueError("Structure must contain 'struct_seq' or 'struct_ids'")

        # Coerce to canonical, comma-separated string when coming from arrays
        if struct_seq_raw is not None:
            struct_seq_str = str(struct_seq_raw)  # handles np.array/list safely
            struct_list = [x.strip() for x in struct_seq_str.split(",") if x.strip()]
        else:
            # struct_ids path
            struct_list = [str(int(x)) for x in structure["struct_ids"]]

        # Build struct text exactly like the working script:
        struct_text = tok.struct_cls_token + "".join(struct_list) + tok.struct_eos_token

        # Build AA text
        if masked_sequence is None:
            # full inverse folding baseline
            L = len(struct_list)
            aa_body = tok.aa_mask_token * L
        else:
            ms = str(masked_sequence)  # <- ensure string
            aa_body = "".join(tok.aa_mask_token if c == "X" else c for c in ms)

        aa_text = tok.aa_cls_token + aa_body + tok.aa_eos_token
        
        # Encode both modalities using tokenizer (string→tokenizer path)
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
        
        # Concatenate structure and AA tokens
        input_tokens = torch.cat([batch_struct["input_ids"], batch_aa["input_ids"]], dim=1).to(self.device)
        
        # Get modality types and non-special masks like generate script
        type_ids = self.model.get_modality_type(input_tokens)
        non_special = self.model.get_non_special_symbol_mask(input_tokens)
        
        # 🔒 Freeze structure tokens and ALL special tokens (crucial for baseline stability)
        struct_type, aa_type = 0, 1
        partial_mask = (type_ids == struct_type)
        partial_mask = partial_mask | (~non_special)  # also freeze ALL special tokens
        
        Ls = batch_struct["input_ids"].shape[1]
        La = batch_aa["input_ids"].shape[1]

        # If baseline: set AA body content to aa_mask_token in the input (like the script)
        if masked_sequence is None:
            input_tokens.masked_fill_((type_ids == aa_type) & non_special, tok._token_to_id[tok.aa_mask_token])
        else:
            # partial masking: freeze non-X positions only
            for i, c in enumerate(ms):
                if c != "X":
                    partial_mask[0, Ls + 1 + i] = True

        # ✅ HARD ASSERTS: structure must be modality 0 end-to-end
        assert torch.all(type_ids[0, :Ls] == 0).item(), \
            "Struct segment contains non-struct tokens (modality != 0). Check struct_text building."

        # Show how many unmasked AA-body positions the diffusion will actually fill
        masked_count = int((~partial_mask).sum().item())
        aa_body_count = int((type_ids[0] == aa_type).sum().item()) - 2  # AA body only
        print(f"   🧮 AA body to diffuse: {aa_body_count}")
        
        print(f"   🧪 struct_tokens: {batch_struct['input_ids'].shape}")
        print(f"   🧪 aa_text: {aa_text[:50]}... (len={len(aa_text)})")
        print(f"   📊 Batch info: input_tokens={input_tokens.shape}, partial_mask={partial_mask.shape}")
        print(f"   📊 Masked positions: {(~partial_mask).sum().item()}")
        
        return {
            "input_tokens": input_tokens,
            "partial_mask": partial_mask
        }

    def _validate_multimodal_batch(self, batch: Dict, target_length: int):
        """Validate multimodal batch construction for debugging."""
        input_tokens = batch["input_tokens"]
        type_ids = batch["type_ids"]
        
        # Confirm modality layout if model supports it
        if hasattr(self.model, "get_modality_type"):
            mt = self.model.get_modality_type(input_tokens)
            struct_correct = (mt[:, :batch["Ls"]] == 0).all().item()
            aa_correct = (mt[:, batch["Ls"]:] == 1).all().item()
            print(f"[Sanity] model modality check: struct={struct_correct}, aa={aa_correct}")
        
        # Verify partial_mask semantics
        partial_mask = batch["partial_mask"]
        struct_frozen = partial_mask[0, :batch["Ls"]].all().item()
        aa_cls_frozen = partial_mask[0, batch["Ls"]].item()
        aa_eos_frozen = partial_mask[0, batch["Ls"] + batch["La"] - 1].item()
        
        print(f"[Sanity] partial_mask validation: struct_frozen={struct_frozen}, "
              f"aa_cls_frozen={aa_cls_frozen}, aa_eos_frozen={aa_eos_frozen}")
    
    def _has_valid_coordinates(self, structure: Dict) -> bool:
        """Check if structure has valid 3D coordinates."""
        # Extract coordinates from structure
        coord_keys = ['coordinates', 'coords', 'xyz', 'backbone_coords', 'atom_positions']
        coords = None
        for key in coord_keys:
            if key in structure and structure[key] is not None:
                coords = structure[key]
                break
        
        if coords is None:
            raise ValueError(f"No coordinates found. Available keys: {list(structure.keys())}")
        
        if hasattr(coords, 'shape') and len(coords.shape) >= 2:
            print(f"   🏗️ Found valid coordinates via '{key}' key, shape: {coords.shape}")
            return True
        elif isinstance(coords, (list, tuple)) and len(coords) > 0:
            print(f"   🏗️ Found valid coordinates via '{key}' key, length: {len(coords)}")
            return True
        else:
            print(f"   ⚠️ No valid coordinates found in structure keys: {list(structure.keys())}")
            return False
    
    def _get_token_id(self, token: str) -> int:
        """Safely get token ID from tokenizer."""
        try:
            if hasattr(self.tokenizer, '_token_to_id') and token in self.tokenizer._token_to_id:
                return self.tokenizer._token_to_id[token]
            elif hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                return self.tokenizer.convert_tokens_to_ids(token)
            elif hasattr(self.tokenizer, 'encode'):
                # Encode single token and get first ID
                encoded = self.tokenizer.encode(token, add_special_tokens=False)
                return encoded[0] if encoded else 0
            else:
                print(f"⚠️  No token mapping method available for token: {token}")
                return 0
        except Exception as e:
            print(f"⚠️  Failed to get token ID for '{token}': {e}")
            return 0
    
    def fill_masked_positions(self, structure: Dict = None, masked_sequence: str = None, 
                             target_length: int = None, temperature: float = 1.0) -> str:
        """
        Fill masked positions using proper structure-conditional generation.
        Supports both baseline (masked_sequence=None) and partial masking.
        """
        # Avoid ambiguous truthiness in fill_masked_positions
        if masked_sequence is not None:
            masked_sequence = str(masked_sequence)
            
        if not target_length:
            target_length = len(masked_sequence) if masked_sequence else 50

        if not self.model or not self.tokenizer:
            raise ValueError("DPLM-2 model not loaded. Cannot perform masked diffusion.")

        # CRITICAL: Use structure-conditional generation - fail fast if no structure
        if structure is None:
            raise ValueError("Need struct_ids/struct_seq or coordinates.")
        if structure.get('struct_ids') is None and structure.get('struct_seq') is None:
            if not self._has_valid_coordinates(structure):
                raise ValueError("Provide struct_ids/struct_seq OR atom_positions [L,37,3].")
            
        print(f"   🏗️ Using structure-conditional generation P(seq|struct)")
        batch = self._create_dplm2_batch(structure, target_length, masked_sequence)

        # Final vocab boundary check before generation
        input_tokens = batch["input_tokens"]
        vocab_n = self.model.net.esm.embeddings.word_embeddings.num_embeddings
        imin, imax = int(input_tokens.min()), int(input_tokens.max())
        
        # Follow exact working approach from generate_dplm2_patched_v2.py
        try:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = self.model.generate(
                    input_tokens=batch["input_tokens"],
                    max_iter=500,  # Increased from 100
                    temperature=0.7,  # Reduced from 1.0
                    unmasking_strategy="deterministic",
                    sampling_strategy="argmax",
                    partial_masks=batch["partial_mask"],  # Use batch partial_mask
                )

                # Decode generated sequence - extract AA tokens by modality exactly like generate script
                generated_tokens = output["output_tokens"][0]  # [T]
                type_ids_out = self.model.get_modality_type(generated_tokens.unsqueeze(0))  # [1,T]
                
                # Extract AA positions (modality type 1) - exactly like generate script
                aa_positions = (type_ids_out[0] == 1).nonzero(as_tuple=False).flatten()
                
                if len(aa_positions) > 0:
                    # Extract AA tokens and decode - use model.tokenizer exactly like generate script
                    aa_tokens = generated_tokens[aa_positions].cpu().tolist()
                    sequence = self.model.tokenizer.decode(aa_tokens)
                    
                    # Remove special tokens - exactly like generate script
                    sequence = sequence.replace(self.model.tokenizer.aa_cls_token, "")
                    sequence = sequence.replace(self.model.tokenizer.aa_eos_token, "")
                    
                    # Filter valid amino acids only - prevent visualization tokens
                    sequence = "".join(c for c in sequence if c in "ACDEFGHIKLMNPQRSTVWY")
                    
                    print(f"[Sanity] decoded AA length: {len(sequence)} (target: {target_length})")
                else:
                    print(f"[Warning] No AA tokens found in generated sequence")
                    # Fallback: try to decode everything and extract valid amino acids
                    decoded = self.model.tokenizer.decode(generated_tokens.cpu().tolist())
                    sequence = "".join(c for c in decoded if c in "ACDEFGHIKLMNPQRSTVWY")
                    print(f"[Fallback] extracted AA sequence length: {len(sequence)}") 
                
                # Handle length validation
                if len(sequence) == target_length:
                    return sequence
                elif len(sequence) > target_length:
                    return sequence[:target_length]
                elif len(sequence) > 0:
                    return sequence
                else:
                    raise ValueError(f"DPLM-2 generated empty sequence after cleaning. Cannot proceed.")

        except Exception as e:
            print(f"Error in DPLM-2 generation: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"DPLM-2 generation failed: {e}. Cannot proceed without a valid sequence.")
    
    def _patch_3b_model_forward(self, model):
        """Patch the 3B model's forward method to handle type_ids parameter."""
        try:
            if hasattr(model, 'forward'):
                original_forward = model.forward
                
                def patched_forward(input_ids=None, attention_mask=None, **kwargs):
                    # Remove type_ids from kwargs if present
                    kwargs.pop('type_ids', None)
                    return original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                
                model.forward = patched_forward
                print(f"   🔧 Applied 3B model forward patch")
            
            # Also patch the network layer's forward method if it exists
            if hasattr(model, 'net') and hasattr(model.net, 'forward'):
                import inspect
                net_forward_signature = inspect.signature(model.net.forward)
                if 'type_ids' not in net_forward_signature.parameters:
                    original_net_forward = model.net.forward
                    
                    def patched_net_forward(*args, **kwargs):
                        # Remove type_ids from kwargs if present
                        kwargs.pop('type_ids', None)
                        return original_net_forward(*args, **kwargs)
                    
                    model.net.forward = patched_net_forward
                    print(f"   🔧 Applied 3B model network forward patch")
                    
        except Exception as e:
            print(f"   ⚠️ Failed to patch 3B model forward: {e}")

    def _with_model(self, expert_name: str):
        """Context manager to temporarily switch to a specific expert model."""
        class ModelContext:
            def __init__(self, integration, expert_name):
                self.integration = integration
                self.expert_name = expert_name
                self.original_model = None
                
            def __enter__(self):
                self.original_model = self.integration.model
                if self.expert_name in self.integration.expert_instances:
                    self.integration.model = self.integration.expert_instances[self.expert_name]
                    print(f"   🔄 Switched to expert: {self.expert_name}")
                    
                    # Apply 3B model patch if needed
                    if '3b' in self.expert_name.lower():
                        self.integration._patch_3b_model_forward(self.integration.model)
                        
                else:
                    print(f"   ⚠️ Expert {self.expert_name} not found, using main model")
                return self.integration.model
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.integration.model = self.original_model
                
        return ModelContext(self, expert_name)

    def generate_baseline_sequence(self, structure: Dict, target_length: int, 
                                 expert_name: str = "airkingbd/dplm2_150m") -> str:
        """Generate baseline sequence using structure-conditional DPLM-2."""
        print(f"   🎯 Generating baseline with {expert_name} (structure-conditional)")
        
        # Create a fully masked sequence for baseline generation
        masked_sequence = 'X' * target_length
        
        # Actually use the specified expert model
        with self._with_model(expert_name):
            return self.fill_masked_positions(structure, masked_sequence, target_length)
    
    def fill_masked_positions_seq_only(self, masked_sequence: str, temperature: float = 0.9,
                                      max_iter: int = 120) -> str:
        """Fill masked positions using sequence-only unmasking (no structure dependency)."""
        if masked_sequence is None:
            raise ValueError("Need a masked_sequence for seq-only unmasking")
        
        masked_sequence = str(masked_sequence)  # Coerce to string
        
        aa_text = self.tokenizer.aa_cls_token + masked_sequence + self.tokenizer.aa_eos_token
        batch_aa = self.tokenizer.batch_encode_plus([aa_text], add_special_tokens=False,
                                                   padding=False, truncation=False, return_tensors="pt")
        aa_ids = batch_aa["input_ids"].to(self.device)  # [1, L]
        
        partial_mask = torch.zeros_like(aa_ids, dtype=torch.bool)
        partial_mask[0, 0] = True   # freeze AA-CLS
        partial_mask[0, -1] = True  # freeze AA-EOS
        
        aa_mask_id = self._get_token_id(self.tokenizer.aa_mask_token)
        
        # Replace 'X' with mask id, freeze others
        for i, ch in enumerate(masked_sequence, start=1):
            if ch == 'X':
                aa_ids[0, i] = aa_mask_id
            else:
                partial_mask[0, i] = True
        
        with torch.no_grad():
            out = self.model.generate(input_tokens=aa_ids,
                                    partial_masks=partial_mask,
                                    max_iter=max_iter,
                                    temperature=temperature)
        
        out_ids = out["output_tokens"][0]
        # strip CLS/EOS, decode, AA-only
        aa_body = out_ids[1:-1]
        seq = self.tokenizer.decode(aa_body, skip_special_tokens=True)
        return "".join(c for c in seq if c in "ACDEFGHIKLMNPQRSTVWY")

    def load_expert_models(self):
        """Load all expert models for multi-expert rollouts."""
        print(" Loading expert models for multi-expert rollouts...")
        
        for expert_name in self.expert_models:
            if expert_name not in self.expert_instances:
                try:
                    expert_model = DPLM2_MODEL_CLASS.from_pretrained(
                        expert_name,
                        from_huggingface=True
                    )
                    expert_model.eval()
                    expert_model.to(self.device)
                    self.expert_instances[expert_name] = expert_model
                    print(f"   ✅ Loaded expert: {expert_name}")
                except Exception as e:
                    print(f"   ❌ Failed to load expert {expert_name}: {e}")
                    # Continue with other experts
        
        print(f"✅ Loaded {len(self.expert_instances)} expert models")
