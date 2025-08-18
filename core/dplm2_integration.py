"""
DPLM-2 Integration Module for MCTS-Guided Inverse Folding

This module provides proper integration with DPLM-2 for:
1. Sequence generation from structure
2. Position-specific amino acid prediction
3. Internal confidence scoring
4. Structure-sequence alignment confidence
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import random
# 🚫 REMOVED: import esm  # ensure ESM symbols exist for byprot internals

# Add the DPLM-2 source code to the path
sys.path.insert(0, '/home/caom/AID3/dplm/src')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel
    from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer
    DPLM2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import DPLM-2 modules: {e}")
    DPLM2_AVAILABLE = False


class DPLM2Integration:
    """
    Integration class for DPLM-2 model for inverse folding tasks.
    Handles model loading, sequence generation, and confidence scoring.
    """
    
    def __init__(self, model_name: str = "airkingbd/dplm2_650m", use_local: bool = False):
        """
        Initialize DPLM-2 integration.
        
        Args:
            model_name: Name of the model to load (HuggingFace model name)
            use_local: Whether to use local model files or download from HuggingFace
        """
        self.model_name = model_name
        self.use_local = use_local
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if DPLM2_AVAILABLE:
            self._load_model()
        else:
            print("Warning: DPLM-2 not available, using fallback methods")
    
    def _load_model(self):
        """Load the DPLM-2 model and tokenizer."""
        try:
            print(f"Loading DPLM-2 model: {self.model_name}")
            # 🚫 REMOVED: ESM loading and patching - not needed for masked diffusion
            
            # Load model from HuggingFace 
            # Support both DPLM-1 and DPLM-2 models to find the working one
            try:
                if "dplm2" in self.model_name:
                    # DPLM-2 models
                    self.model = MultimodalDiffusionProteinLanguageModel.from_pretrained(
                        self.model_name,
                        from_huggingface=True
                    )
                    print(f"✅ Loaded DPLM-2 model: {self.model_name}")
                else:
                    # DPLM-1 models - try loading directly
                    from byprot.models.dplm.dplm import DiffusionProteinLanguageModel
                    self.model = DiffusionProteinLanguageModel.from_pretrained(
                        self.model_name,
                        from_huggingface=True
                    )
                    print(f"✅ Loaded DPLM-1 model: {self.model_name}")
            except Exception as e:
                print(f"❌ Failed to load {self.model_name}: {e}")
                # Fallback to DPLM-2 loading
                self.model = MultimodalDiffusionProteinLanguageModel.from_pretrained(
                    self.model_name,
                    from_huggingface=True
                )
            
            # Configure for inverse folding task
            # Set the model to inverse folding mode (structure -> sequence)
            self.model.eval()  # Always in eval mode for inference
            
            # Get tokenizer from model (THIS IS THE CORRECT TOKENIZER!)
            self.tokenizer = self.model.tokenizer
            print(f"✅ Loaded tokenizer: {type(self.tokenizer).__name__}")
            print(f"✅ Has aa_mask_token: {hasattr(self.tokenizer, 'aa_mask_token')}")
            
            # Set up tokenizer attributes with fallbacks
            if not hasattr(self.tokenizer, 'aa_mask_token'):
                if hasattr(self.tokenizer, 'mask_token'):
                    self.tokenizer.aa_mask_token = self.tokenizer.mask_token
                elif hasattr(self.tokenizer, 'pad_token'):
                    self.tokenizer.aa_mask_token = self.tokenizer.pad_token
                else:
                    self.tokenizer.aa_mask_token = "<mask>"
            
            if not hasattr(self.tokenizer, 'aa_cls_token'):
                if hasattr(self.tokenizer, 'cls_token'):
                    self.tokenizer.aa_cls_token = self.tokenizer.cls_token
                else:
                    self.tokenizer.aa_cls_token = "<s>"
            
            if not hasattr(self.tokenizer, 'aa_eos_token'):
                if hasattr(self.tokenizer, 'eos_token'):
                    self.tokenizer.aa_eos_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.aa_eos_token = "</s>"
            
            if not hasattr(self.tokenizer, 'struct_cls_token'):
                self.tokenizer.struct_cls_token = "<struct>"
            
            if not hasattr(self.tokenizer, 'struct_eos_token'):
                self.tokenizer.struct_eos_token = "</struct>"
            
            # Create token_to_id mapping if needed
            if not hasattr(self.tokenizer, '_token_to_id'):
                if hasattr(self.tokenizer, 'token_to_id'):
                    self.tokenizer._token_to_id = self.tokenizer.token_to_id
                elif hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                    self.tokenizer._token_to_id = {}
                    for token_name in ['aa_mask_token', 'aa_cls_token', 'aa_eos_token', 'struct_cls_token', 'struct_eos_token']:
                        if hasattr(self.tokenizer, token_name):
                            token_value = getattr(self.tokenizer, token_name)
                            try:
                                token_id = self.tokenizer.convert_tokens_to_ids(token_value)
                                self.tokenizer._token_to_id[token_value] = token_id
                            except Exception:
                                self.tokenizer._token_to_id[token_value] = 0
                else:
                    self.tokenizer._token_to_id = {
                        getattr(self.tokenizer, 'aa_mask_token', '<mask>'): 0,
                        getattr(self.tokenizer, 'aa_cls_token', '<s>'): 1,
                        getattr(self.tokenizer, 'aa_eos_token', '</s>'): 2,
                        getattr(self.tokenizer, 'struct_cls_token', '<struct>'): 3,
                        getattr(self.tokenizer, 'struct_eos_token', '</struct>'): 4,
                    }
            
            # Load structure tokenizer for coordinate encoding
            print("Loading structure tokenizer...")
            try:
                from byprot.models.utils import get_struct_tokenizer
                # Try loading without triggering ESM contact regression download
                self.struct_tokenizer = get_struct_tokenizer("airkingbd/struct_tokenizer", eval_mode=True)
                self.struct_tokenizer = self.struct_tokenizer.to(self.device)
                self.struct_tokenizer.eval()
                print("✅ Structure tokenizer loaded successfully")
            except Exception as e:
                print(f"⚠️  Structure tokenizer failed to load: {e}")
                print("Will use fallback structure tokens (AAR will be lower)")
                self.struct_tokenizer = None
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"Successfully loaded DPLM-2 model on {self.device}")
            
        except Exception as e:
            print(f"Error loading DPLM-2 model: {e}")
            print("Using fallback random generation methods")
            self.model = None
            self.tokenizer = None
            self.struct_tokenizer = None
    
    def _get_token_id(self, token: str) -> int:
        """
        Safely get token ID from tokenizer.
        
        Args:
            token: Token string to convert to ID
            
        Returns:
            Token ID (integer)
        """
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
    
    def _coordinates_to_structure_tokens(self, structure: Dict) -> str:
        """Convert 3D coordinates to structure tokens using the structure tokenizer."""
        if not hasattr(self, 'struct_tokenizer') or self.struct_tokenizer is None:
            # 🚫 NO FALLBACK: Structure tokenizer is required for proper operation
            raise ValueError("Structure tokenizer not available. Cannot convert coordinates to structure tokens.")
        
        try:
            # Extract coordinates from structure
            if 'coordinates' in structure:
                coords = structure['coordinates']  # Shape: [L, 3, 3] for N, CA, C atoms
            elif 'backbone_coords' in structure:
                coords = structure['backbone_coords']
            else:
                raise ValueError("No coordinates found in structure")
            
            # Convert numpy to torch if needed
            import torch
            import numpy as np
            if isinstance(coords, np.ndarray):
                coords = torch.tensor(coords, dtype=torch.float32)
            
            # Handle different coordinate formats
            if coords.ndim == 3:  # [L, 3, 3] - backbone atoms (N, CA, C)
                seq_length = coords.shape[0]
                
                # Create full atom representation (37 atoms per residue)
                all_atom_positions = torch.zeros(1, seq_length, 37, 3, dtype=torch.float32)
                
                # Copy backbone atoms to their correct positions
                all_atom_positions[0, :, 0, :] = coords[:, 0, :]  # N atoms
                all_atom_positions[0, :, 1, :] = coords[:, 1, :]  # CA atoms  
                all_atom_positions[0, :, 2, :] = coords[:, 2, :]  # C atoms
                
                # Use CA positions as approximation for side chain atoms
                for atom_idx in range(3, 37):
                    all_atom_positions[0, :, atom_idx, :] = coords[:, 1, :]
                
            elif coords.ndim == 2:  # [L, 3] - single atom per residue (probably CA)
                seq_length = coords.shape[0]
                
                # Create full atom representation with CA positions
                all_atom_positions = torch.zeros(1, seq_length, 37, 3, dtype=torch.float32)
                all_atom_positions[0, :, 1, :] = coords  # CA atoms at position 1
                
                # Use CA positions as approximation for other atoms
                for atom_idx in range(37):
                    if atom_idx != 1:
                        all_atom_positions[0, :, atom_idx, :] = coords
                        
            else:
                raise ValueError(f"Unexpected coordinate format: {coords.shape}")
            
            # Create residue mask and move to device
            res_mask = torch.ones(1, seq_length, dtype=torch.float32)
            seq_length_tensor = torch.tensor([seq_length])
            
            all_atom_positions = all_atom_positions.to(self.device)
            res_mask = res_mask.to(self.device)
            seq_length_tensor = seq_length_tensor.to(self.device)
            
            # Tokenize structure using DPLM-2's structure tokenizer
            with torch.no_grad():
                tokens = self.struct_tokenizer.tokenize(
                    all_atom_positions, res_mask, seq_length_tensor
                )
            
            # Convert to string format for DPLM-2
            struct_tokens = tokens[0].tolist()
            struct_text = ",".join(map(str, struct_tokens))
            
            return struct_text
            
        except Exception as e:
            print(f"❌ Structure tokenization failed: {e}")
            import traceback
            traceback.print_exc()
            # 🚫 NO FALLBACK: If structure tokenization fails, we cannot proceed
            raise ValueError(f"Structure tokenization failed: {e}. Cannot proceed without proper structure tokens.")
    
    def create_inverse_folding_input(self, structure: Dict, target_length: int) -> Dict:
        """
        Create input for inverse folding task.
        
        Args:
            structure: Dictionary containing structure information
            target_length: Target sequence length
            
        Returns:
            Dictionary with tokenized input for DPLM-2
        """
        if not self.tokenizer:
            raise ValueError("DPLM-2 tokenizer not loaded. Cannot create input.")
        
        try:
            # For inverse folding: AA sequence should be all mask tokens
            aa_sequence = self.tokenizer.aa_mask_token * target_length
            aa_text = self.tokenizer.aa_cls_token + aa_sequence + self.tokenizer.aa_eos_token
            
            # Structure tokens: use REAL structure tokens from 3D coordinates
            struct_sequence = self._coordinates_to_structure_tokens(structure)
            struct_text = self.tokenizer.struct_cls_token + struct_sequence + self.tokenizer.struct_eos_token
            

            
            # Tokenize using the same format as working generate_dplm2.py
            # Create lists like the working script does
            struct_list = [struct_text]
            aa_list = [aa_text]
            
            batch_struct = self.tokenizer.batch_encode_plus(
                struct_list,
                add_special_tokens=False,
                padding="longest",
                truncation=True,
                max_length=2048,  # Increase max length for structure tokens
                return_tensors="pt"
            )
            
            batch_aa = self.tokenizer.batch_encode_plus(
                aa_list,
                add_special_tokens=False,
                padding="longest",
                truncation=True,
                max_length=1024,  # Reasonable max length for AA sequence
                return_tensors="pt"
            )
            
            # Combine structure and amino acid tokens
            input_tokens = torch.concat(
                [batch_struct["input_ids"], batch_aa["input_ids"]], dim=1
            )
            input_tokens = input_tokens.to(self.device)
            
            # Get type IDs and masks
            type_ids = self.model.get_modality_type(input_tokens)
            non_special = self.model.get_non_special_symbol_mask(input_tokens)
            
            # Check for None returns and handle gracefully
            if type_ids is None:
                print("Warning: get_modality_type returned None, using fallback")
                type_ids = torch.zeros_like(input_tokens)
            if non_special is None:
                print("Warning: get_non_special_symbol_mask returned None, using fallback")
                non_special = torch.ones_like(input_tokens, dtype=torch.bool)
            
            # Ensure all tensors are on the same device
            type_ids = type_ids.to(self.device)
            non_special = non_special.to(self.device)
            
            # Mask amino acid tokens for inverse folding
            aa_type = 1
            input_tokens.masked_fill_(
                (type_ids == aa_type) & non_special,
                self._get_token_id(self.tokenizer.aa_mask_token)
            )
            
            return {
                "input_tokens": input_tokens,
                "type_ids": type_ids,
                "non_special": non_special
            }
            
        except Exception as e:
            print(f"Error creating inverse folding input: {e}")
            raise ValueError(f"DPLM-2 input creation failed: {e}. Cannot proceed without proper input.")
    
    def _create_masked_input(self, masked_sequence: str, structure: Dict, target_length: int) -> Dict:
        """
        Create input with masked sequence for DPLM-2.
        
        Args:
            masked_sequence: Sequence with 'X' for masked positions
            structure: Structure information
            target_length: Target sequence length
            
        Returns:
            Dictionary with tokenized input for DPLM-2
        """
        if not self.tokenizer:
            raise ValueError("DPLM-2 tokenizer not loaded. Cannot create input.")
        
        try:
            # Convert masked sequence to proper format with special tokens
            aa_text = self.tokenizer.aa_cls_token + masked_sequence + self.tokenizer.aa_eos_token
            
            # Structure tokens: use REAL structure tokens from 3D coordinates
            struct_sequence = self._coordinates_to_structure_tokens(structure)
            struct_text = self.tokenizer.struct_cls_token + struct_sequence + self.tokenizer.struct_eos_token
            
            # Tokenize using the same format as working generate_dplm2.py
            # Create lists like the working script does
            struct_list = [struct_text]
            aa_list = [aa_text]
            
            batch_struct = self.tokenizer.batch_encode_plus(
                struct_list,
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt"
            )
            
            batch_aa = self.tokenizer.batch_encode_plus(
                aa_list,
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt"
            )
            
            # Combine structure and amino acid tokens
            input_tokens = torch.concat(
                [batch_struct["input_ids"], batch_aa["input_ids"]], dim=1
            )
            input_tokens = input_tokens.to(self.device)
            
            # Get type IDs and masks
            type_ids = self.model.get_modality_type(input_tokens)
            non_special = self.model.get_non_special_symbol_mask(input_tokens)
            
            # Check for None returns and handle gracefully
            if type_ids is None:
                print("Warning: get_modality_type returned None, using fallback")
                type_ids = torch.zeros_like(input_tokens)
            if non_special is None:
                print("Warning: get_non_special_symbol_mask returned None, using fallback")
                non_special = torch.ones_like(input_tokens, dtype=torch.bool)
            
            # Ensure all tensors are on the same device
            type_ids = type_ids.to(self.device)
            non_special = non_special.to(self.device)
            
            return {
                "input_tokens": input_tokens,
                "type_ids": type_ids,
                "non_special": non_special
            }
            
        except Exception as e:
            print(f"Error creating masked input: {e}")
            raise ValueError(f"DPLM-2 input creation failed: {e}. Cannot proceed without proper input.")
    
    def _create_structure_tokens_from_structure(self, structure: Dict, target_length: int) -> str:
        """
        Create structure tokens from structure information.
        This is a placeholder - in real implementation would convert 3D coordinates to tokens.
        """
        # Placeholder: create random structure tokens
        # In real implementation, this would use the structure tokenizer
        struct_vocab_size = 8192
        tokens = []
        for _ in range(target_length):
            token_id = random.randint(0, struct_vocab_size - 1)
            tokens.append(str(token_id))
        return ",".join(tokens)
    
    def _create_fallback_input(self, target_length: int) -> Dict:
        """
        🚫 REMOVED: Random fallback input creation.
        
        This method has been removed because:
        1. Random inputs provide no meaningful generation
        2. All inputs should be properly tokenized for DPLM-2
        3. Fallback inputs mislead model behavior
        
        If DPLM-2 tokenization fails, the system should fail gracefully rather than generate random data.
        """
        raise NotImplementedError(
            "Random fallback input creation has been removed. "
            "All inputs must be properly tokenized for DPLM-2 model. "
            "If DPLM-2 tokenization fails, the system should fail gracefully rather than generate random data."
        )
    
    def _create_dplm2_batch(self, structure: Dict, target_length: int, masked_sequence: str = None) -> Dict:
        """
        Create DPLM-2 batch format for sequence generation.
        
        This handles both modes:
        1. Structure-conditional: P(seq|struct) when structure is available
        2. Unconditional: P(seq) when structure is missing (just sequence generation)
        """
        try:
            # Handle both structure-conditional and unconditional modes
            if masked_sequence:
                aa_tokens = masked_sequence
            else:
                aa_tokens = self.tokenizer.aa_mask_token * target_length
            
            # Create amino acid text
            aa_text = self.tokenizer.aa_cls_token + aa_tokens + self.tokenizer.aa_eos_token
            aa_text_spaced = aa_text.replace(',', ' ')
            
            # MODE 1: Structure-conditional generation (P(seq|struct))
            if structure and self._has_valid_coordinates(structure):
                try:
                    # Get structure tokens from coordinates
                    struct_tokens = self._coordinates_to_structure_tokens(structure)
                    struct_text = self.tokenizer.struct_cls_token + struct_tokens + self.tokenizer.struct_eos_token
                    struct_text_spaced = struct_text.replace(',', ' ')
                    
                    # Tokenize structure with proper limits
                    batch_struct = self.tokenizer.batch_encode_plus(
                        [struct_text_spaced],
                        add_special_tokens=False,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    
                    # Tokenize AA sequence
                    batch_aa = self.tokenizer.batch_encode_plus(
                        [aa_text_spaced],
                        add_special_tokens=False,
                        padding=True,
                        truncation=True,
                        max_length=target_length + 10,
                        return_tensors="pt",
                    )
                    
                    # Combine structure + AA tokens
                    input_tokens = torch.concat(
                        [batch_struct["input_ids"], batch_aa["input_ids"]], dim=1
                    )
                    input_tokens = input_tokens.to(self.device)
                    
                    # Get type IDs and masks
                    aa_type = 1
                    struct_type = 0
                    non_special = self.model.get_non_special_symbol_mask(input_tokens)
                    type_ids = self.model.get_modality_type(input_tokens)
                    
                    # Mask only the positions that are actually masked in the sequence
                    if masked_sequence:
                        masked_positions = []
                        for i, char in enumerate(masked_sequence):
                            if char == 'X':
                                token_pos = i + 1  # +1 for CLS token
                                masked_positions.append(token_pos)
                        
                        for pos in masked_positions:
                            if pos < input_tokens.shape[1]:
                                mask_token_id = self._get_token_id(self.tokenizer.aa_mask_token)
                                input_tokens[0, pos] = mask_token_id
                    else:
                        # Mask all AA tokens
                        mask_token_id = self._get_token_id(self.tokenizer.aa_mask_token)
                        input_tokens.masked_fill_(
                            (type_ids == aa_type) & non_special,
                            mask_token_id,
                        )
                    
                    # Create batch for structure-conditional generation
                    batch = {
                        "input_tokens": input_tokens,
                        "partial_mask": type_ids == struct_type
                    }
                    
                    return batch
                    
                except Exception as e:
                    print(f"Structure-conditional mode failed: {e}, falling back to unconditional...")
            
            # MODE 2: Unconditional sequence generation (P(seq))
            try:
                # Tokenize only the AA sequence (no structure)
                batch_aa = self.tokenizer.batch_encode_plus(
                    [aa_text_spaced],
                    add_special_tokens=False,
                    padding=True,
                    truncation=True,
                    max_length=target_length + 10,
                    return_tensors="pt",
                )
                
                input_tokens = batch_aa["input_ids"].to(self.device)
                
                # Ensure input token length matches expected sequence length
                if input_tokens.shape[1] > target_length + 20:
                    input_tokens = input_tokens[:, :target_length + 20]
                
                # Mask the positions that are actually masked
                if masked_sequence:
                    masked_positions = []
                    for i, char in enumerate(masked_sequence):
                        if char == 'X':
                            token_pos = i + 1  # +1 for CLS token
                            masked_positions.append(token_pos)
                    
                    for pos in masked_positions:
                        if pos < input_tokens.shape[1]:
                            mask_token_id = self._get_token_id(self.tokenizer.aa_mask_token)
                            input_tokens[0, pos] = mask_token_id
                else:
                    # Mask all positions
                    mask_token_id = self._get_token_id(self.tokenizer.aa_mask_token)
                    eos_token_id = self._get_token_id(self.tokenizer.aa_eos_token)
                    input_tokens.masked_fill_(
                        input_tokens != eos_token_id,
                        mask_token_id,
                    )
                
                # Create batch for unconditional generation
                batch = {
                    "input_tokens": input_tokens,
                    "partial_mask": None  # No structure conditioning
                }
                
                return batch
                
            except Exception as e:
                print(f"Unconditional mode failed: {e}")
                return None
            
        except Exception as e:
            print(f"❌ Batch creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _has_valid_coordinates(self, structure: Dict) -> bool:
        """Check if structure has valid 3D coordinates."""
        try:
            if 'coordinates' in structure and structure['coordinates']:
                return True
            if 'coords' in structure and structure['coords']:
                return True
            if 'xyz' in structure and structure['xyz']:
                return True
            return False
        except:
            return False

    def generate_sequence(self, structure: Dict, target_length: int, 
                         max_iter: int = 100, temperature: float = 1.0, 
                         masked_sequence: str = None) -> str:
        """
        Generate sequence using DPLM-2 in both modes:
        
        1. Structure-conditional: P(seq|struct) when structure is available
        2. Unconditional: P(seq) when structure is missing (just sequence generation)
        
        Args:
            structure: Structure information (optional)
            target_length: Target sequence length
            max_iter: Maximum generation iterations
            temperature: Sampling temperature
            masked_sequence: Sequence with some positions masked (X tokens)
        """
        if not self.model:
            raise ValueError("DPLM-2 model not loaded. Cannot generate sequence.")
        
        try:
            # Create batch (handles both modes automatically)
            batch = self._create_dplm2_batch(structure, target_length, masked_sequence)
            
            if not batch:
                raise ValueError("Failed to create DPLM-2 batch. Cannot proceed with generation.")
            
            # Generate using DPLM-2 API
            with torch.no_grad():
                if batch.get("partial_mask") is not None:
                    # Structure-conditional generation
                    output = self.model.generate(
                        input_tokens=batch["input_tokens"],
                        max_iter=max_iter,
                        temperature=temperature,
                        partial_masks=batch["partial_mask"],
                    )
                else:
                    # Unconditional generation (no structure)
                    output = self.model.generate(
                        input_tokens=batch["input_tokens"],
                        max_iter=max_iter,
                        temperature=temperature,
                    )
            
            # Decode the generated sequence
            output_tokens = output["output_tokens"]
            decoded_sequences = self.tokenizer.batch_decode(
                output_tokens, skip_special_tokens=False
            )
            
            # Process the decoded sequence
            if decoded_sequences and len(decoded_sequences) > 0:
                generated_sequence = decoded_sequences[0]
                
                # Extract amino acid sequence (remove special tokens)
                aa_start = generated_sequence.find(self.tokenizer.aa_cls_token)
                aa_end = generated_sequence.find(self.tokenizer.aa_eos_token)
                
                # Handle missing end token gracefully
                if aa_start != -1:
                    if aa_end != -1:
                        # Normal case: both tokens found
                        aa_sequence = generated_sequence[aa_start + len(self.tokenizer.aa_cls_token):aa_end]
                    else:
                        # End token not found, extract from start to end of string
                        aa_sequence = generated_sequence[aa_start + len(self.tokenizer.aa_cls_token):]
                        
                        # Try to find alternative end tokens
                        alternative_end_tokens = ['</s>', '<eos>', '<end>', 'EOS', 'END']
                        for alt_token in alternative_end_tokens:
                            alt_end = generated_sequence.find(alt_token)
                            if alt_end != -1:
                                aa_sequence = generated_sequence[aa_start + len(self.tokenizer.aa_cls_token):alt_end]
                                break
                else:
                    # Start token not found, try to extract without it
                    cleaned_sequence = generated_sequence
                    for token in [self.tokenizer.aa_cls_token, self.tokenizer.aa_eos_token, '<s>', '</s>', '<cls>', '<eos>']:
                        cleaned_sequence = cleaned_sequence.replace(token, '')
                    
                    aa_sequence = cleaned_sequence
                
                # Clean up the sequence (remove any remaining special tokens and non-amino acid characters)
                cleaned_chars = []
                for c in aa_sequence:
                    if c in "ACDEFGHIKLMNPQRSTVWY":
                        cleaned_chars.append(c)
                    elif c in "X":  # Keep mask tokens
                        cleaned_chars.append(c)
                    elif c.isalpha():  # Keep other letters (might be valid)
                        cleaned_chars.append(c)
                    # Skip numbers, punctuation, and other special characters
                
                aa_sequence = ''.join(cleaned_chars)
                
                # Handle length validation
                if len(aa_sequence) == target_length:
                    return aa_sequence
                elif len(aa_sequence) > target_length:
                    return aa_sequence[:target_length]
                elif len(aa_sequence) > 0:
                    return aa_sequence
                else:
                    raise ValueError(f"DPLM-2 generated empty sequence after cleaning. Cannot proceed.")
            
            # Fallback if decoding failed
            raise ValueError("DPLM-2 generation failed to produce a valid sequence.")
            
        except Exception as e:
            print(f"Error in DPLM-2 generation: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"DPLM-2 generation failed: {e}. Cannot proceed without a valid sequence.")

    def _generate_fallback_sequence(self, target_length: int) -> str:
        """
        🚫 REMOVED: Random fallback sequence generation.
        
        This method has been removed because:
        1. Random sequences provide no meaningful optimization
        2. All sequence generation should use DPLM-2 model
        3. Fallback sequences mislead MCTS evaluation
        
        If DPLM-2 fails, the system should fail gracefully rather than generate random data.
        """
        raise NotImplementedError(
            "Random fallback sequence generation has been removed. "
            "All sequences must be generated by DPLM-2 model. "
            "If DPLM-2 fails, the system should fail gracefully rather than generate random data."
        )
    
    def get_position_confidence(self, sequence: str, position: int) -> float:
        """
        🚫 REMOVED: Fake random confidence calculation.
        
        This method has been removed because:
        1. Random confidence scores provide no meaningful information
        2. All confidence scoring should use real pLDDT computation
        3. Fake scores mislead MCTS optimization decisions
        
        Use real_plddt_computation.compute_plddt_from_structure instead.
        """
        raise NotImplementedError(
            "Fake random confidence calculation has been removed. "
            "Use utils.real_plddt_computation.compute_plddt_from_structure instead. "
            "This provides real structural quality assessment based on 3D coordinates."
        )
    
    def get_attention_confidence(self, sequence: str) -> List[float]:
        """
        🚫 REMOVED: Fake random attention confidence calculation.
        
        This method has been removed because:
        1. Random attention scores provide no meaningful information
        2. All confidence scoring should use real pLDDT computation
        3. Fake scores mislead MCTS optimization decisions
        
        Use real_plddt_computation.compute_plddt_from_structure instead.
        """
        raise NotImplementedError(
            "Fake random attention confidence calculation has been removed. "
            "Use utils.real_plddt_computation.compute_plddt_from_structure instead. "
            "This provides real structural quality assessment based on 3D coordinates."
        )
    
    def get_diffusion_confidence(self, sequence: str) -> List[float]:
        """
        🚫 REMOVED: Fake random diffusion confidence calculation.
        
        This method has been removed because:
        1. Random diffusion scores provide no meaningful information
        2. All confidence scoring should use real pLDDT computation
        3. Fake scores mislead MCTS optimization decisions
        
        Use real_plddt_computation.compute_plddt_from_structure instead.
        """
        raise NotImplementedError(
            "Fake random diffusion confidence calculation has been removed. "
            "Use utils.real_plddt_computation.compute_plddt_from_structure instead. "
            "This provides real structural quality assessment based on 3D coordinates."
        )
    
    def get_comprehensive_confidence(self, sequence: str) -> List[float]:
        """
        🚫 REMOVED: Fake random comprehensive confidence calculation.
        
        This method has been removed because:
        1. Random comprehensive scores provide no meaningful information
        2. All confidence scoring should use real pLDDT computation
        3. Fake scores mislead MCTS optimization decisions
        
        Use real_plddt_computation.compute_plddt_from_structure instead.
        """
        raise NotImplementedError(
            "Fake random comprehensive confidence calculation has been removed. "
            "Use utils.real_plddt_computation.compute_plddt_from_structure instead. "
            "This provides real structural quality assessment based on 3D coordinates."
        )
    
    def is_available(self) -> bool:
        """Check if DPLM-2 model is available and loaded."""
        return self.model is not None and self.tokenizer is not None

    def fill_masked_positions(self, structure: Dict = None, masked_sequence: str = None, 
                             target_length: int = None, temperature: float = 1.0) -> str:
        """
        🎯 MASKED DIFFUSION: Fill masked positions in an existing sequence using DPLM-2.
        
        This method implements proper diffusion unmasking using DPLM-2's partial_masks:
        1. Take a sequence with X positions (e.g., "MTGIGLHTAMWAEDDDVPGTEAAVARAIEYDVDFXEIPMLDPPXIDTAYF...")
        2. Use DPLM-2 to fill ONLY the X positions while preserving unmasked positions
        3. Return the sequence with X's filled (e.g., "MTGIGLHTAMWAEDDDVPGTEAAVARAIEYDVDFAEIPMLDPPAIDTAYF...")
        
        This leverages DPLM-2's motif scaffolding approach for sequence-to-sequence diffusion.
        
        Args:
            structure: Structure information for conditioning (can be None for pure diffusion)
            masked_sequence: Sequence with some positions masked (X tokens)
            target_length: Target sequence length (if None, use sequence length)
            temperature: Sampling temperature for generation
            
        Returns:
            Completed sequence with all masked positions filled
        """
        if not target_length:
            target_length = len(masked_sequence)
        
        if not masked_sequence:
            raise ValueError("Masked sequence is required for diffusion unmasking")
        
        if not self.model or not self.tokenizer:
            raise ValueError("DPLM-2 model not loaded. Cannot perform masked diffusion.")
        
        try:
            # Use DPLM-2's partial_masks for proper masked diffusion
            # Create the input sequence with proper special tokens
            aa_text = self.tokenizer.aa_cls_token + masked_sequence + self.tokenizer.aa_eos_token
            
            # Tokenize the amino acid sequence
            batch_aa = self.tokenizer.batch_encode_plus(
                [aa_text],
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=target_length + 10,
                return_tensors="pt",
            )
            
            input_tokens = batch_aa["input_ids"].to(self.device)
            
            # Create partial_masks to control which positions are filled
            # partial_masks[i] = True means position i should be preserved (NOT filled)
            # partial_masks[i] = False means position i should be filled by diffusion
            
            # Initialize all positions as fillable (False)
            partial_masks = torch.zeros_like(input_tokens, dtype=torch.bool)
            
            # Find which positions in the input_tokens correspond to unmasked positions in the sequence
            for i, char in enumerate(masked_sequence):
                if char != 'X':
                    # This position should be preserved - mark it as True in partial_masks
                    token_pos = i + 1  # +1 for CLS token
                    if token_pos < input_tokens.shape[1]:
                        partial_masks[0, token_pos] = True
            
            # Also preserve special tokens (CLS, EOS)
            partial_masks[0, 0] = True  # CLS token
            if input_tokens.shape[1] > 1:
                partial_masks[0, -1] = True  # EOS token (if present)
            
            # Generate using DPLM-2 with partial_masks for masked diffusion
            with torch.no_grad():
                output = self.model.generate(
                    input_tokens=input_tokens,
                    max_iter=100,  # Reasonable number of diffusion steps
                    temperature=temperature,
                    partial_masks=partial_masks,  # KEY: This controls what gets filled
                    sampling_strategy="annealing@2.0:1.0"
                )
            
            # Decode the generated sequence
            output_tokens = output["output_tokens"]
            decoded_sequences = self.tokenizer.batch_decode(
                output_tokens, skip_special_tokens=False
            )
            
            if decoded_sequences and len(decoded_sequences) > 0:
                completed_sequence = decoded_sequences[0]
                
                # Extract amino acid sequence (remove special tokens)
                aa_start = completed_sequence.find(self.tokenizer.aa_cls_token)
                aa_end = completed_sequence.find(self.tokenizer.aa_eos_token)
                
                if aa_start != -1:
                    if aa_end != -1:
                        # Normal case: both tokens found
                        aa_sequence = completed_sequence[aa_start + len(self.tokenizer.aa_cls_token):aa_end]
                    else:
                        # End token not found, extract from start to end of string
                        aa_sequence = completed_sequence[aa_start + len(self.tokenizer.aa_cls_token):]
                        
                        # Try to find alternative end tokens
                        alternative_end_tokens = ['</s>', '<eos>', '<end>', 'EOS', 'END']
                        for alt_token in alternative_end_tokens:
                            alt_end = completed_sequence.find(alt_token)
                            if alt_end != -1:
                                aa_sequence = completed_sequence[aa_start + len(self.tokenizer.aa_cls_token):alt_end]
                                break
                else:
                    # Start token not found, try to extract without it
                    cleaned_sequence = completed_sequence
                    for token in [self.tokenizer.aa_cls_token, self.tokenizer.aa_eos_token, '<s>', '</s>', '<cls>', '<eos>']:
                        cleaned_sequence = cleaned_sequence.replace(token, '')
                    aa_sequence = cleaned_sequence
                
                # Clean up the sequence
                cleaned_chars = []
                for c in aa_sequence:
                    if c in "ACDEFGHIKLMNPQRSTVWY":
                        cleaned_chars.append(c)
                    elif c.isalpha():  # Keep other letters (might be valid)
                        cleaned_chars.append(c)
                    # Skip numbers, punctuation, and other special characters
                
                aa_sequence = ''.join(cleaned_chars)
                
                # Handle length validation
                if len(aa_sequence) == target_length:
                    pass  # Perfect length
                elif len(aa_sequence) > target_length:
                    aa_sequence = aa_sequence[:target_length]
                elif len(aa_sequence) > 0:
                    pass  # Return as-is
                else:
                    raise ValueError(f"DPLM-2 generated empty sequence after cleaning. Cannot proceed.")
                
                return aa_sequence
            else:
                raise ValueError("DPLM-2 generation failed to produce a valid sequence.")
                
        except Exception as e:
            print(f"Error in DPLM-2 masked diffusion: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"DPLM-2 masked diffusion failed: {e}. Cannot proceed without a valid sequence.")
    
    def apply_md4_style_transitions(self, sequence: str, num_transitions: int = 2, 
                                   temperature: float = 0.8) -> str:
        """
        🎯 MD4-STYLE TRANSITIONS: Apply small transitions from unmask to mask to avoid propagating errors.
        
        This method implements the MD4 innovation:
        1. Take a completed sequence
        2. Mask a small number of previously unmasked positions
        3. Use DPLM-2 to fill these new masked positions
        4. This prevents error propagation and enables exploration
        
        Args:
            sequence: Completed sequence to apply transitions to
            num_transitions: Number of positions to mask/unmask (default: 2)
            temperature: Sampling temperature for generation
            
        Returns:
            Sequence with new positions filled by DPLM-2
        """
        try:
            # 🎯 STRATEGY: Mask some previously unmasked positions for exploration
            # This prevents getting stuck in local optima and propagates errors
            
            # Choose random positions to mask (avoiding any existing masked positions)
            sequence_length = len(sequence)
            positions_to_mask = random.sample(range(sequence_length), min(num_transitions, sequence_length))
            
            # Create sequence with new positions masked
            masked_sequence = list(sequence)
            for pos in positions_to_mask:
                masked_sequence[pos] = 'X'
            masked_sequence = ''.join(masked_sequence)
            
            print(f"🎯 MD4-style transition: masking {len(positions_to_mask)} new positions for exploration")
            print(f"   Positions masked: {positions_to_mask}")
            print(f"   Masked sequence: {masked_sequence[:50]}...")
            
            # Use DPLM-2 to fill these new masked positions
            completed_sequence = self.fill_masked_positions(
                structure=None,  # No structure for pure diffusion
                masked_sequence=masked_sequence,
                target_length=sequence_length,
                temperature=temperature
            )
            
            if completed_sequence and len(completed_sequence) == sequence_length:
                print(f"🎯 MD4-style transition successful: {len(positions_to_mask)} new positions filled")
                return completed_sequence
            else:
                print(f"❌ MD4-style transition failed: could not complete sequence")
                return sequence  # Return original if transition fails
                
        except Exception as e:
            print(f"⚠️ MD4-style transition failed: {e}, returning original sequence")
            return sequence
    
    def _fill_masked_positions_fallback(self, masked_sequence: str, target_length: int) -> str:
        """
        🚫 REMOVED: Random amino acid fallback for masked positions.
        
        This method has been removed because:
        1. Random amino acids provide no meaningful optimization
        2. All amino acid selection should use DPLM-2 model
        3. Fallback amino acids mislead MCTS evaluation
        
        If DPLM-2 fails, the system should fail gracefully rather than generate random data.
        """
        raise NotImplementedError(
            "Random amino acid fallback for masked positions has been removed. "
            "All amino acid selection must use DPLM-2 model. "
            "If DPLM-2 fails, the system should fail gracefully rather than generate random data."
        )

    def _simple_random_fill_masked_positions(self, masked_sequence: str, target_length: int) -> Dict:
        """
        🎯 SIMPLE RANDOM AMINO ACID FILLING: Fill masked positions with random amino acids.
        
        This is a practical solution when DPLM-2's masked diffusion is not available:
        1. Take the original sequence with X positions
        2. Fill each X with a random amino acid from the standard set
        3. Preserve all unmasked positions exactly as they are
        4. Let MCTS optimize the random fills through its search process
        
        Args:
            masked_sequence: Sequence with 'X' for masked positions
            target_length: Expected sequence length
            
        Returns:
            Dictionary with filled sequence tokens
        """
        try:
            print(f"🎯 Simple random filling:")
            print(f"   Masked sequence: {masked_sequence[:50]}...")
            print(f"   Target length: {target_length}")
            print(f"   Masked positions: {masked_sequence.count('X')}")
            
            # 🎯 SIMPLE APPROACH: Fill each X with a random amino acid
            filled_sequence = ""
            for i, char in enumerate(masked_sequence):
                if char == 'X':
                    # Fill masked position with random amino acid
                    random_aa = random.choice("ACDEFGHIKLMNPQRSTVWY")
                    filled_sequence += random_aa
                    print(f"   Position {i}: X -> {random_aa}")
                else:
                    # Preserve original amino acid
                    filled_sequence += char
            
            # Verify length
            if len(filled_sequence) != target_length:
                print(f"⚠️  Length mismatch: {len(filled_sequence)} != {target_length}")
                # Truncate or pad to target length
                if len(filled_sequence) > target_length:
                    filled_sequence = filled_sequence[:target_length]
                else:
                    filled_sequence = filled_sequence + 'X' * (target_length - len(filled_sequence))
                print(f"🎯 Corrected length: {len(filled_sequence)}")
            
            print(f"🎯 Random filling completed:")
            print(f"   Original masked: {masked_sequence[:50]}...")
            print(f"   Filled sequence: {filled_sequence[:50]}...")
            print(f"   Final length: {len(filled_sequence)}")
            print(f"   Remaining X's: {filled_sequence.count('X')}")
            
            # Create output dictionary that matches expected format
            combined_text = self.tokenizer.aa_cls_token + filled_sequence + self.tokenizer.aa_eos_token
            combined_tokens = self.tokenizer.encode(combined_text, add_special_tokens=False)
            combined_tokens = torch.tensor([combined_tokens], device=self.device)
            
            mock_output = {
                "output_tokens": combined_tokens
            }
            
            print(f"🎯 Simple random filling successful!")
            return mock_output
            
        except Exception as e:
            print(f"❌ Simple random filling failed: {e}")
            return None


def test_dplm2_integration():
    """Test the DPLM-2 integration."""
    from utils.protein_utils import create_mock_structure_no_sequence
    
    # Create test structure
    structure = create_mock_structure_no_sequence(length=50)
    
    # Initialize integration
    dplm2 = DPLM2Integration(use_local=False)
    
    print(f"DPLM-2 available: {dplm2.is_available()}")
    
    # Test sequence generation
    sequence = dplm2.generate_sequence(structure, target_length=50)
    print(f"Generated sequence: {sequence}")
    print(f"Sequence length: {len(sequence)}")
    
    # Test confidence scoring
    if dplm2.is_available():
        confidences = dplm2.get_comprehensive_confidence(sequence)
        print(f"Confidence scores: {confidences[:5]}...")  # First 5 positions
    
    return sequence


if __name__ == "__main__":
    test_dplm2_integration() 