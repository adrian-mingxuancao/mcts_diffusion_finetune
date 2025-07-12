"""
DPLM-2 Integration Module for MCTS-Guided Inverse Folding

This module provides proper integration with DPLM-2 for:
1. Sequence generation from structure
2. Position-specific amino acid prediction
3. Internal confidence scoring
4. Structure-sequence alignment confidence
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import sys
import os

# Add the src directory to the path to import DPLM-2 modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel as DPLM2
from byprot.models.dplm2 import DPLM2Bit
from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DPLM2Integration:
    """
    Integration class for DPLM-2 model in MCTS-guided inverse folding.
    """
    
    def __init__(self, model_name: str = "airkingbd/dplm2_650m", device: str = "auto", use_bit_model: bool = False, use_local: bool = True):
        """
        Initialize DPLM-2 integration.
        
        Args:
            model_name: Model name (HuggingFace name or local checkpoint path)
            device: Device to load model on ("auto", "cpu", "cuda")
            use_bit_model: Whether to use DPLM-2 bit model
            use_local: Whether to use local models instead of downloading from HuggingFace
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bit_model = use_bit_model
        self.use_local = use_local
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        logger.info(f"Initializing DPLM-2 integration with model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Using bit model: {use_bit_model}")
        logger.info(f"Using local models: {use_local}")
    
    def load_model(self):
        """Load DPLM-2 model and tokenizer."""
        try:
            logger.info("Loading DPLM-2 model and tokenizer...")
            
            # Load the appropriate DPLM-2 model
            if self.use_bit_model:
                self.model = DPLM2Bit.from_pretrained(
                    self.model_name, 
                    from_huggingface=not self.use_local
                )
            else:
                self.model = DPLM2.from_pretrained(
                    self.model_name, 
                    from_huggingface=not self.use_local
                )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Get the tokenizer from the model
            self.tokenizer = self.model.tokenizer
            
            self.is_loaded = True
            logger.info("DPLM-2 model loaded successfully!")
            logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
            
        except Exception as e:
            logger.error(f"Error loading DPLM-2 model: {e}")
            self.is_loaded = False
            raise
    
    def encode_structure(self, structure_tokens: str) -> torch.Tensor:
        """
        Encode structure tokens for DPLM-2 input.
        
        Args:
            structure_tokens: Structure tokens as string
            
        Returns:
            Encoded structure tokens
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Add special tokens for structure
        struct_tokens = (
            self.tokenizer.struct_cls_token + 
            structure_tokens + 
            self.tokenizer.struct_eos_token
        )
        
        # Encode
        encoded = self.tokenizer.encode_plus(
            struct_tokens,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        return encoded["input_ids"].to(self.device)
    
    def create_inverse_folding_input(self, structure_tokens: str, sequence_length: int) -> Dict:
        """
        Create input for inverse folding task.
        
        Args:
            structure_tokens: Structure tokens as string
            sequence_length: Length of the target sequence
            
        Returns:
            Input batch for DPLM-2
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Create masked sequence
        masked_sequence = self.tokenizer.aa_mask_token * sequence_length
        masked_sequence = (
            self.tokenizer.aa_cls_token + 
            masked_sequence + 
            self.tokenizer.aa_eos_token
        )
        
        # Encode structure and sequence
        struct_encoded = self.tokenizer.encode_plus(
            self.tokenizer.struct_cls_token + structure_tokens + self.tokenizer.struct_eos_token,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        seq_encoded = self.tokenizer.encode_plus(
            masked_sequence,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        # Concatenate structure and sequence tokens
        input_tokens = torch.concat([struct_encoded["input_ids"], seq_encoded["input_ids"]], dim=1)
        input_tokens = input_tokens.to(self.device)
        
        # Get modality types
        type_ids = self.model.get_modality_type(input_tokens)
        non_special = self.model.get_non_special_symbol_mask(input_tokens)
        
        # Mask amino acid tokens for inverse folding
        aa_type = 1
        input_tokens.masked_fill_(
            (type_ids == aa_type) & non_special,
            self.tokenizer._token_to_id[self.tokenizer.aa_mask_token]
        )
        
        # Create batch
        batch = {
            "input_tokens": input_tokens,
            "partial_mask": type_ids == aa_type
        }
        
        return batch
    
    def generate_sequence(self, structure_tokens: str, max_iter: int = 100, temperature: float = 1.0) -> Tuple[str, List[float]]:
        """
        Generate sequence from structure using DPLM-2.
        
        Args:
            structure_tokens: Structure tokens as string
            max_iter: Maximum generation iterations
            temperature: Sampling temperature
            
        Returns:
            Generated sequence and confidence scores
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Estimate sequence length from structure tokens
        sequence_length = len(structure_tokens.split(',')) if ',' in structure_tokens else len(structure_tokens)
        
        # Create input batch
        batch = self.create_inverse_folding_input(structure_tokens, sequence_length)
        
        # Generate
        with torch.no_grad():
            output_tokens, output_scores = self.model.generate(
                input_tokens=batch["input_tokens"],
                max_iter=max_iter,
                temperature=temperature,
                partial_masks=batch["partial_mask"],
                unmasking_strategy="stochastic1.0",
                sampling_strategy="annealing@2.2:1.0"
            )
        
        # Extract amino acid tokens (second half of the output)
        type_ids = self.model.get_modality_type(output_tokens)
        aa_mask = type_ids == 1  # amino acid type
        
        # Get amino acid tokens and scores
        aa_tokens = output_tokens[aa_mask]
        aa_scores = output_scores[aa_mask]
        
        # Convert tokens to sequence
        sequence = self.tokenizer.decode(aa_tokens, skip_special_tokens=True)
        
        # Convert scores to confidence
        confidence_scores = aa_scores.cpu().numpy().tolist()
        
        return sequence, confidence_scores
    
    def get_position_confidence(self, structure_tokens: str, position: int, max_iter: int = 50) -> Dict[str, float]:
        """
        Get confidence for a specific position in the sequence.
        
        Args:
            structure_tokens: Structure tokens as string
            position: Position in the sequence (0-indexed)
            max_iter: Maximum generation iterations
            
        Returns:
            Dictionary with amino acid probabilities and confidence
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Create input for generation
        sequence_length = len(structure_tokens.split(',')) if ',' in structure_tokens else len(structure_tokens)
        batch = self.create_inverse_folding_input(structure_tokens, sequence_length)
        
        # Run a few steps to get logits for the specific position
        with torch.no_grad():
            # Initialize output tokens
            output_tokens, output_scores = self.model.initialize_output_tokens(
                batch["input_tokens"], partial_masks=batch["partial_mask"]
            )
            
            # Run decoder for a few steps
            decoder_out = {
                "output_tokens": output_tokens,
                "output_scores": output_scores,
                "step": 0,
                "max_step": max_iter,
                "history": [output_tokens.clone()]
            }
            
            # Get logits for the position
            net_out = self.model.net(input_ids=output_tokens)
            logits = net_out["logits"]
            
            # Get amino acid logits for the specific position
            type_ids = self.model.get_modality_type(output_tokens)
            aa_mask = type_ids == 1  # amino acid type
            
            if position < aa_mask.sum():
                # Find the actual position in the amino acid tokens
                aa_positions = torch.where(aa_mask)[0]
                if position < len(aa_positions):
                    pos_idx = aa_positions[position]
                    aa_logits = logits[0, pos_idx, :]  # [vocab_size]
                    
                    # Get probabilities for amino acid tokens only
                    aa_token_ids = []
                    for i, token in enumerate(self.tokenizer.all_tokens):
                        if token in ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']:
                            aa_token_ids.append(i)
                    
                    aa_probs = torch.softmax(aa_logits[aa_token_ids], dim=-1)
                    
                    # Create result dictionary
                    result = {}
                    for i, token_id in enumerate(aa_token_ids):
                        token = self.tokenizer.all_tokens[token_id]
                        result[token] = aa_probs[i].item()
                    
                    # Add confidence (max probability)
                    result['confidence'] = aa_probs.max().item()
                    
                    return result
        
        return {'confidence': 0.0}
    
    def get_attention_confidence(self, structure_tokens: str, position: int, max_iter: int = 50) -> float:
        """
        Get attention-based confidence for a position.
        
        Args:
            structure_tokens: Structure tokens as string
            position: Position in the sequence
            max_iter: Maximum generation iterations
            
        Returns:
            Attention confidence score
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Create input for generation
        sequence_length = len(structure_tokens.split(',')) if ',' in structure_tokens else len(structure_tokens)
        batch = self.create_inverse_folding_input(structure_tokens, sequence_length)
        
        # Run decoder with attention weights
        with torch.no_grad():
            output_tokens, output_scores = self.model.initialize_output_tokens(
                batch["input_tokens"], partial_masks=batch["partial_mask"]
            )
            
            decoder_out = {
                "output_tokens": output_tokens,
                "output_scores": output_scores,
                "step": 0,
                "max_step": max_iter,
                "history": [output_tokens.clone()]
            }
            
            # Get attention weights
            decoder_result = self.model.forward_decoder(
                decoder_out, need_attn_weights=True, partial_masks=batch["partial_mask"]
            )
            
            if 'attentions' in decoder_result and decoder_result['attentions'] is not None:
                # Calculate attention confidence (average attention weight)
                attentions = decoder_result['attentions']  # [num_layers, batch, heads, seq_len, seq_len]
                avg_attention = attentions.mean(dim=(0, 2))  # Average over layers and heads
                
                # Get attention for the specific position
                type_ids = self.model.get_modality_type(output_tokens)
                aa_mask = type_ids == 1
                
                if position < aa_mask.sum():
                    aa_positions = torch.where(aa_mask)[0]
                    if position < len(aa_positions):
                        pos_idx = aa_positions[position]
                        # Average attention from this position to all structure tokens
                        struct_mask = type_ids == 0
                        struct_attention = avg_attention[0, pos_idx, struct_mask].mean().item()
                        return struct_attention
        
        return 0.0
    
    def get_diffusion_confidence(self, structure_tokens: str, position: int, timestep: int = 100) -> float:
        """
        Get diffusion-based confidence for a position.
        
        Args:
            structure_tokens: Structure tokens as string
            position: Position in the sequence
            timestep: Diffusion timestep for confidence estimation
            
        Returns:
            Diffusion confidence score
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Create input for generation
        sequence_length = len(structure_tokens.split(',')) if ',' in structure_tokens else len(structure_tokens)
        batch = self.create_inverse_folding_input(structure_tokens, sequence_length)
        
        with torch.no_grad():
            # Get noise level at the specified timestep
            t = torch.tensor([timestep], device=self.device)
            
            # Create noisy tokens
            output_tokens, _ = self.model.initialize_output_tokens(
                batch["input_tokens"], partial_masks=batch["partial_mask"]
            )
            
            # Add noise
            type_ids = self.model.get_modality_type(output_tokens)
            maskable_mask = self.model.get_non_special_symbol_mask(output_tokens, batch["partial_mask"])
            
            noisy_tokens = self.model.q_sample(output_tokens, t, type_ids, maskable_mask)
            
            # Get model prediction
            net_out = self.model.net(input_ids=noisy_tokens)
            logits = net_out["logits"]
            
            # Calculate confidence based on prediction certainty
            type_ids = self.model.get_modality_type(noisy_tokens)
            aa_mask = type_ids == 1
            
            if position < aa_mask.sum():
                aa_positions = torch.where(aa_mask)[0]
                if position < len(aa_positions):
                    pos_idx = aa_positions[position]
                    pos_logits = logits[0, pos_idx, :]
                    
                    # Calculate entropy-based confidence
                    probs = torch.softmax(pos_logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum()
                    max_entropy = torch.log(torch.tensor(len(probs), dtype=torch.float))
                    confidence = 1.0 - (entropy / max_entropy).item()
                    
                    return max(0.0, min(1.0, confidence))
        
        return 0.0
    
    def get_comprehensive_confidence(self, structure_tokens: str, position: int) -> Dict[str, float]:
        """
        Get comprehensive confidence scores for a position.
        
        Args:
            structure_tokens: Structure tokens as string
            position: Position in the sequence
            
        Returns:
            Dictionary with different confidence measures
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Get different types of confidence
        token_confidence = self.get_position_confidence(structure_tokens, position)
        attention_confidence = self.get_attention_confidence(structure_tokens, position)
        diffusion_confidence = self.get_diffusion_confidence(structure_tokens, position)
        
        return {
            'token_confidence': token_confidence.get('confidence', 0.0),
            'attention_confidence': attention_confidence,
            'diffusion_confidence': diffusion_confidence,
            'combined_confidence': (token_confidence.get('confidence', 0.0) + 
                                  attention_confidence + 
                                  diffusion_confidence) / 3.0
        }
    
    def decode_tokens_to_sequence(self, tokens: torch.Tensor) -> str:
        """
        Decode tokens to amino acid sequence.
        
        Args:
            tokens: Token tensor
            
        Returns:
            Amino acid sequence
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Filter to amino acid tokens only
        type_ids = self.model.get_modality_type(tokens)
        aa_mask = type_ids == 1
        
        aa_tokens = tokens[aa_mask]
        sequence = self.tokenizer.decode(aa_tokens, skip_special_tokens=True)
        
        return sequence


def test_dplm2_integration():
    """Test the DPLM-2 integration."""
    print("Testing DPLM-2 integration...")
    
    # Initialize integration
    dplm2 = DPLM2Integration(model_name="airkingbd/dplm2_650m", use_bit_model=False)
    
    try:
        # Load model
        dplm2.load_model()
        print("✓ Model loaded successfully")
        
        # Test with a simple structure
        test_structure = "A" * 10  # Simple structure tokens
        print(f"Testing with structure: {test_structure}")
        
        # Generate sequence
        sequence, scores = dplm2.generate_sequence(test_structure, max_iter=20, temperature=1.0)
        print(f"Generated sequence: {sequence}")
        print(f"Average confidence: {np.mean(scores):.3f}")
        
        # Get position confidence
        if len(sequence) > 0:
            pos_confidence = dplm2.get_position_confidence(test_structure, 0)
            print(f"Position 0 confidence: {pos_confidence}")
        
        print("✓ DPLM-2 integration test completed successfully!")
        
    except Exception as e:
        print(f"✗ DPLM-2 integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dplm2_integration() 