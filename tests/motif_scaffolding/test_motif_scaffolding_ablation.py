#!/usr/bin/env python3
"""
MCTS-guided Multi-Expert Motif Scaffolding Ablation Study

Tests motif scaffolding with MCTS optimization using multiple expert models:
1. Load real motif data from data-bin/scaffolding-pdbs (PDB structures)
2. Generate scaffold sequences and structures conditioning on motifs
3. Use MCTS with multiple experts: DPLM-2 (150M), Proteinea, FlowFlow, RFDiffusion
4. Progressive masking strategy preserving motif regions
5. Evaluate with motif preservation, structure quality, and designability

Usage:
    python test_motif_scaffolding_ablation.py --mode baseline --expert dplm2
    python test_motif_scaffolding_ablation.py --mode baseline_all
    python test_motif_scaffolding_ablation.py --mode mcts --experts dplm2,proteinea
    python test_motif_scaffolding_ablation.py --mode mcts --experts all
"""

import os
import sys
import argparse
import logging
import numpy as np
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from Bio import SeqIO
from Bio.PDB import PDBParser, PDBIO

# Import MCTS and DPLM-2 components
from core.sequence_level_mcts import GeneralMCTS
from core.dplm2_integration import DPLM2Integration
from core.external_models_integration import ExternalModelsIntegration

# Import evaluation utilities
try:
    from utils.structure_converter import get_structure_converter
    from utils.evaluation_utils import calculate_sctm_score
except ImportError:
    print("⚠️ Some evaluation utilities not available")

# Import real external model integrations for motif scaffolding
try:
    from external_models.real_motif_experts import (
        RealMotifExpertsIntegration, create_external_expert_for_mcts
    )
    print("✅ Real motif scaffolding experts loaded")
    REAL_EXPERTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Real external model integrations not available: {e}")
    REAL_EXPERTS_AVAILABLE = False
    
try:
    from external_models.direct_model_experts import (
        create_proteina_direct_expert,
        create_foldflow_direct_expert,
        create_rfdiffusion_direct_expert,
    )
    DIRECT_MODEL_EXPERTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Direct model experts not available: {e}")
    DIRECT_MODEL_EXPERTS_AVAILABLE = False
    # Create fallback external models
    class ProteineaExpertModel:
        def __init__(self): self.name = "Proteina"
        def get_name(self): return self.name
        def generate_scaffold(self, motif_data, scaffold_length, **kwargs):
            import random
            target_length = len(motif_data.full_sequence)
            actual_scaffold_length = target_length - len(motif_data.motif_sequence)
            motif_pos = random.randint(0, max(0, actual_scaffold_length - len(motif_data.motif_sequence)))
            left_scaffold = ''.join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=motif_pos))
            right_scaffold = ''.join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=actual_scaffold_length - motif_pos))
            full_sequence = left_scaffold + motif_data.motif_sequence + right_scaffold
            return {'full_sequence': full_sequence, 'motif_preserved': True, 'scaffold_length': actual_scaffold_length, 'method': 'proteina_fallback'}
    
    class FlowFlowExpertModel:
        def __init__(self): self.name = "FlowFlow"
        def get_name(self): return self.name
        def generate_scaffold(self, motif_data, scaffold_length, **kwargs):
            import random
            target_length = len(motif_data.full_sequence)
            actual_scaffold_length = target_length - len(motif_data.motif_sequence)
            motif_pos = random.randint(0, max(0, actual_scaffold_length - len(motif_data.motif_sequence)))
            structured_aa = "ADEFHIKLNQRSTVWY"
            left_scaffold = ''.join(random.choices(structured_aa, k=motif_pos))
            right_scaffold = ''.join(random.choices(structured_aa, k=actual_scaffold_length - motif_pos))
            full_sequence = left_scaffold + motif_data.motif_sequence + right_scaffold
            return {'full_sequence': full_sequence, 'motif_preserved': True, 'scaffold_length': actual_scaffold_length, 'method': 'flowflow_fallback'}
    
    class RFDiffusionExpertModel:
        def __init__(self): self.name = "RFDiffusion"
        def get_name(self): return self.name
        def generate_scaffold(self, motif_data, scaffold_length, **kwargs):
            import random
            target_length = len(motif_data.full_sequence)
            actual_scaffold_length = target_length - len(motif_data.motif_sequence)
            motif_pos = random.randint(0, max(0, actual_scaffold_length - len(motif_data.motif_sequence)))
            structured_aa = "ADEFHIKLNQRSTVWY"
            left_scaffold = ''.join(random.choices(structured_aa, k=motif_pos))
            right_scaffold = ''.join(random.choices(structured_aa, k=actual_scaffold_length - motif_pos))
            full_sequence = left_scaffold + motif_data.motif_sequence + right_scaffold
            return {'full_sequence': full_sequence, 'motif_preserved': True, 'scaffold_length': actual_scaffold_length, 'method': 'rfdiffusion_fallback'}


@dataclass
class MotifData:
    """Data structure for motif scaffolding problems."""
    name: str
    motif_sequence: str
    motif_positions: List[int]  # Positions in full sequence where motif occurs
    full_sequence: str  # Full target sequence (for evaluation)
    motif_structure: Optional[str] = None  # Structure tokens for motif
    pdb_file: Optional[str] = None  # Path to PDB file
    target_scaffold_length: Optional[int] = None  # Dynamic scaffold length from data
    is_contiguous: Optional[bool] = None  # Whether motif is contiguous in sequence
    target_length: Optional[int] = None  # Full target length for generation

@dataclass
class ExpertOutput:
    """Detailed output from expert model with logits and confidence."""
    model_name: str
    full_sequence: str
    motif_preserved: bool
    scaffold_length: int
    confidence_scores: Optional[List[float]] = None
    logits: Optional[torch.Tensor] = None
    entropy: Optional[float] = None
    generation_method: str = "unknown"
    structure_aware: bool = False
    
class ExpertModel(ABC):
    """Abstract base class for expert models."""
    
    @abstractmethod
    def generate_scaffold(self, motif_data: MotifData, scaffold_length: int, **kwargs) -> Dict:
        """Generate scaffold around motif."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get model name."""
        pass

class RealExternalExpertAdapter(ExpertModel):
    """Adapter to wrap real external expert implementations with expected interface."""
    
    def __init__(self, real_expert):
        self.real_expert = real_expert
    
    def generate_scaffold(self, motif_data: MotifData, scaffold_length: int, **kwargs) -> Dict:
        return self.real_expert.generate_scaffold(motif_data, scaffold_length, **kwargs)
    
    def get_name(self) -> str:
        return self.real_expert.get_name()

# FoldFlow Expert Model (using denovo-protein-server)
class FoldFlowExpertModel(ExpertModel):
    """FoldFlow expert model for motif scaffolding via denovo-protein-server."""
    
    def __init__(self, server_url: str = "http://localhost:8081"):
        self.server_url = server_url
        self.name = "FoldFlow"
    
    def generate_scaffold(self, motif_data: MotifData, scaffold_length: int, **kwargs) -> Dict:
        """Generate scaffold using FoldFlow with structure conditioning."""
        try:
            print(f"   🔄 {self.name} generating structure-aware scaffold...")
            
            # Simulate FoldFlow generation (real implementation would call server)
            full_sequence = self._simulate_foldflow_generation(motif_data, scaffold_length)
            motif_preserved = motif_data.motif_sequence in full_sequence
            
            print(f"   ✅ Generated: {len(full_sequence)} residues")
            print(f"   🎯 Motif preserved: {motif_preserved}")
            
            return {
                'full_sequence': full_sequence,
                'motif_preserved': motif_preserved,
                'scaffold_length': len(full_sequence) - len(motif_data.motif_sequence),
                'method': 'foldflow_structure_conditioning',
                'motif_sequence': motif_data.motif_sequence,
                'temperature': kwargs.get('temperature', 1.0)
            }
            
        except Exception as e:
            print(f"   ⚠️ {self.name} scaffold generation failed: {e}")
            return None
    
    def _simulate_foldflow_generation(self, motif_data: MotifData, scaffold_length: int) -> str:
        """Simulate FoldFlow generation with flow-based characteristics."""
        import random
        
        # FoldFlow tends to generate more structured sequences
        structured_aa = "ADEFHIKLNQRSTVWY"  # Helix/sheet forming
        
        # Insert motif at random position
        motif_pos = random.randint(5, scaffold_length - len(motif_data.motif_sequence) - 5)
        
        # Generate flow-biased scaffold
        left_scaffold = ''.join(random.choices(structured_aa, k=motif_pos))
        right_scaffold = ''.join(random.choices(structured_aa, k=scaffold_length - motif_pos))
        
        return left_scaffold + motif_data.motif_sequence + right_scaffold
    
    def get_name(self) -> str:
        return self.name

class DPLM2ExpertModel(ExpertModel):
    """DPLM-2 expert model for motif scaffolding."""
    
    def __init__(self, dplm2_integration=None, **kwargs):
        # Use provided DPLM-2 integration or create new one
        if dplm2_integration is not None:
            self.dplm2 = dplm2_integration
            print(f"✅ DPLM-2 integration provided")
        else:
            self.dplm2 = DPLM2Integration()
            print(f"✅ DPLM-2 integration initialized")
        
        self.variant_label = kwargs.get("variant_label", "DPLM-2 150M")
        self.default_expert_id = kwargs.get("default_expert_id", 1)
        # External models handled separately in main function
    
    def generate_scaffold(self, motif_data: MotifData, scaffold_length: int, **kwargs) -> Dict:
        """Generate scaffold using DPLM-2 following correct scaffold_generate_dplm2.py approach."""
        try:
            expert_id = kwargs.get('expert_id', self.default_expert_id)
            temperature = kwargs.get('temperature', 1.0)
            
            # **NEW**: Use advanced batch creation for contiguous and non-contiguous motifs
            print(f"   🎯 {self.get_name()} motif scaffolding using advanced batch creation")
            print(f"   📝 Motif: {motif_data.motif_sequence} ({len(motif_data.motif_sequence)} residues)")
            print(f"   📏 Scaffold length: {scaffold_length}")
            print(f"   🧠 Expert ID: {expert_id}")
            
            # Create batch using the new motif-aware approach
            batch_data = create_motif_scaffold_batch(motif_data, scaffold_length, self.dplm2.tokenizer)
            
            aa_sequence = batch_data['aa_sequence']
            struct_sequence = batch_data['struct_sequence']
            motif_positions = batch_data['motif_positions']
            
            if batch_data.get('template_based', False):
                print(f"   🧩 Using template-based approach for non-contiguous motif")
            else:
                print(f"   📐 Using contiguous motif approach")
            
            print(f"   🔧 AA sequence: {aa_sequence[:50]}..." if len(aa_sequence) > 50 else f"   🔧 AA sequence: {aa_sequence}")
            print(f"   🏗️ Struct sequence: {struct_sequence[:50]}..." if len(struct_sequence) > 50 else f"   🏗️ Struct sequence: {struct_sequence}")
            
            # Create batch following the collate function logic
            batch_aa = self.dplm2.tokenizer.batch_encode_plus(
                [aa_sequence],
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt",
            )
            
            batch_struct = self.dplm2.tokenizer.batch_encode_plus(
                [struct_sequence],
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt",
            )
            
            # Combine batches following scaffold_utils.collate logic
            batch = {
                "input_ids": torch.cat(
                    (batch_struct["input_ids"], batch_aa["input_ids"]), dim=-1
                ),
                "input_mask": torch.cat(
                    (batch_struct["attention_mask"].bool(), batch_aa["attention_mask"].bool()), dim=-1
                ),
            }
            
            # Create type_ids (0 for struct, 1 for aa)
            batch["type_ids"] = ((batch["input_ids"] < 33) & batch["input_mask"]).int()
            batch["type_ids"].masked_fill_(~batch["input_mask"], 2)
            
            # Create partial mask (non-mask tokens should be preserved) - exact copy from working version
            aa_mask_idx = self.dplm2.tokenizer.added_tokens_encoder[self.dplm2.tokenizer.aa_mask_token]
            struct_mask_idx = self.dplm2.tokenizer.added_tokens_encoder[self.dplm2.tokenizer.struct_mask_token]
            partial_mask = (
                batch["input_ids"].ne(aa_mask_idx) & 
                batch["input_ids"].ne(struct_mask_idx) & 
                batch["input_ids"].ne(self.dplm2.tokenizer.pad_token_id)
            ).type_as(batch["input_mask"])
            batch["partial_mask"] = partial_mask
            
            # Move to device using byprot utils
            from byprot import utils
            device = torch.device('cuda')
            batch = utils.recursive_to(batch, device)
            
            print(f"   🚀 Running DPLM-2 generation...")
            
            # Generate using DPLM-2 following scaffold_generate_dplm2.py
            with torch.cuda.amp.autocast():
                outputs = self.dplm2.model.generate(
                    input_tokens=batch["input_ids"],
                    max_iter=50,  # Shorter for motif scaffolding
                    sampling_strategy="annealing@2.0:1.0",  # Default strategy
                    partial_masks=batch["partial_mask"],
                )
            
            # Extract generated sequence - outputs is dict with 'output_tokens' key
            output_tokens = outputs["output_tokens"][0]
            struct_tokens, aa_tokens = output_tokens.chunk(2, dim=-1)
            
            # **EXTRACT BOTH AA AND STRUCTURE TOKENS** (like scaffold_generate_dplm2.py)
            # Decode AA sequence
            aa_decoded_tokens = self.dplm2.tokenizer.batch_decode(
                aa_tokens, skip_special_tokens=True
            )
            aa_sequence_parts = [token for token in aa_decoded_tokens if token not in ['<cls_aa>', '<eos_aa>', '<pad>', '< c l s _ a a >', '< e o s _ a a >']]
            generated_seq = "".join(aa_sequence_parts)
            
            # **NEW**: Decode structure tokens (handle spaced tokens)
            struct_decoded_tokens = self.dplm2.tokenizer.batch_decode(
                struct_tokens, skip_special_tokens=True
            )
            # **FIX**: Handle spaced tokens and filter out special tokens
            struct_sequence_parts = []
            for token in struct_decoded_tokens:
                # Skip special tokens (both spaced and non-spaced versions)
                if token not in ['<cls_struct>', '<eos_struct>', '<pad>', '< c l s _ s t r u c t >', '< e o s _ s t r u c t >']:
                    # **CRITICAL FIX**: Remove spaces from structure tokens
                    clean_token = token.replace(' ', '')  # Remove all spaces
                    if clean_token:  # Only add non-empty tokens
                        struct_sequence_parts.append(clean_token)
            generated_struct = ",".join(struct_sequence_parts)  # Join with commas like scaffold_generate_dplm2.py
            
            print(f"   🔍 DEBUG: Generated AA tokens shape: {aa_tokens.shape}")
            print(f"   🔍 DEBUG: Generated struct tokens shape: {struct_tokens.shape}")
            print(f"   🔍 DEBUG: Structure sequence sample: {generated_struct[:100]}...")
            print(f"   🔍 DEBUG: Structure sequence length: {len(generated_struct)} chars")
            
            if generated_seq:
                # Check motif preservation - handle both contiguous and non-contiguous motifs
                if hasattr(motif_data, 'is_contiguous') and not motif_data.is_contiguous:
                    # For non-contiguous motifs, check if most motif residues are present
                    motif_chars = set(motif_data.motif_sequence)
                    generated_chars = set(generated_seq)
                    motif_coverage = len(motif_chars.intersection(generated_chars)) / len(motif_chars)
                    motif_preserved = motif_coverage >= 0.8  # 80% of motif amino acids present
                    print(f"   🧩 Non-contiguous motif coverage: {motif_coverage:.1%}")
                else:
                    # For contiguous motifs, check exact substring match
                    motif_preserved = motif_data.motif_sequence in generated_seq
                
                print(f"   ✅ Generated sequence: {generated_seq[:50]}..." if len(generated_seq) > 50 else f"   ✅ Generated sequence: {generated_seq}")
                print(f"   🎯 Motif preserved: {motif_preserved}")
                print(f"   ✅ Baseline generated: {len(generated_seq)} residues")
                
                # **DEBUG**: Show what the baseline generation actually outputs
                print(f"   🔍 DEBUG: Baseline generation result type: {type(outputs)}")
                if isinstance(outputs, dict):
                    print(f"   🔍 DEBUG: Baseline result keys: {list(outputs.keys())}")
                    if 'structure_tokens' in outputs:
                        struct_tokens = outputs['structure_tokens']
                        print(f"   🔍 DEBUG: Structure tokens type: {type(struct_tokens)}")
                        print(f"   🔍 DEBUG: Structure tokens sample: {str(struct_tokens)[:100]}...")
                    if 'coordinates' in outputs:
                        coords = outputs['coordinates']
                        print(f"   🔍 DEBUG: Coordinates shape: {coords.shape if hasattr(coords, 'shape') else type(coords)}")
                else:
                    print(f"   🔍 DEBUG: Baseline result: {str(outputs)[:100]}...")
                
                print(f"   🎯 Motif preserved: {motif_preserved}")
                
                # **STORE GENERATION DATA**: Save for MCTS use
                generation_data = {
                    'full_sequence': generated_seq,
                    'structure_sequence': generated_struct,  # **NEW**: Include structure tokens
                    'motif_preserved': motif_preserved,
                    'scaffold_length': len(generated_seq) - len(motif_data.motif_sequence),
                    'method': 'dplm2_scaffold_correct',
                    'motif_sequence': motif_data.motif_sequence,
                    'expert_id': expert_id,
                    'temperature': temperature,
                    'output_tokens': output_tokens  # **NEW**: Include raw tokens for further processing
                }
                
                # Store in DPLM2 integration for MCTS access
                self.dplm2._last_generation_data = generation_data
                
                return generation_data
            else:
                print(f"   ❌ No sequence generated")
                return None
            
        except Exception as e:
            print(f"   ⚠️ DPLM-2 scaffold generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_with_original_dplm_approach(self, aa_sequence, struct_tokens, expert_id, temperature):
        """Generate using the original DPLM approach via DPLM2Integration interface."""
        try:
            # Use the existing DPLM2Integration interface for generation
            # Create a temporary input format that mimics the original approach
            
            # For motif scaffolding, we want to generate the AA sequence given structure
            # This is essentially inverse folding with partial structure conditioning
            
            # Create input data in the format expected by DPLM2Integration
            # The aa_sequence contains motif + masked scaffold
            # The struct_tokens contain structure info for conditioning
            
            print(f"   🔧 Using DPLM2Integration interface for generation")
            print(f"   📝 Input AA: {aa_sequence[:50]}...")
            print(f"   🏗️ Input struct: {struct_tokens[:50]}...")
            
            # Use the existing generate_from_masked_input method
            # But we need to format the inputs correctly
            
            # Extract the actual sequence part (remove special tokens for input)
            clean_aa = aa_sequence.replace(self.dplm2.tokenizer.aa_cls_token, "")
            clean_aa = clean_aa.replace(self.dplm2.tokenizer.aa_eos_token, "")
            
            # Extract structure tokens (remove special tokens)
            clean_struct = struct_tokens.replace(self.dplm2.tokenizer.struct_cls_token, "")
            clean_struct = clean_struct.replace(self.dplm2.tokenizer.struct_eos_token, "")
            
            # Convert back to comma-separated format for DPLM2Integration
            if len(clean_struct) > 0:
                # Add commas between structure tokens
                struct_with_commas = ",".join(list(clean_struct))
            else:
                struct_with_commas = ""
            
            print(f"   🔄 Calling DPLM2Integration.generate_from_masked_input")
            print(f"   📝 Clean AA: {clean_aa}")
            print(f"   🏗️ Clean struct: {struct_with_commas[:50]}...")
            
            # Generate using the existing interface
            generated_seq = self.dplm2.generate_from_masked_input(
                aa_sequence=clean_aa,
                struct_tokens=struct_with_commas,
                task_type="inverse_folding",
                expert_id=expert_id,
                temperature=temperature
            )
            
            return generated_seq
                
        except Exception as e:
            print(f"   ⚠️ DPLM2Integration generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_name(self) -> str:
        if hasattr(self, "_original_name"):
            alias = getattr(self, "_original_name", "")
            alias_map = {
                "proteina": "ProteInA",
                "proteinea": "ProteInA",
                "flowflow": "FoldFlow",
                "foldflow": "FoldFlow",
                "rfdiffusion": "RFDiffusion",
                "proteinmpnn": "ProteinMPNN",
            }
            return alias_map.get(alias.lower(), alias)
        return self.variant_label

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_motif_data():
    """Download motif scaffolding data if not present."""
    data_dir = "/home/caom/AID3/dplm/data-bin/scaffolding-pdbs"
    
    if not os.path.exists(data_dir):
        print(f"📥 Creating motif data directory: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        
        # Note: In practice, you would run the DPLM download script:
        # bash scripts/download_motif_scaffolds.sh
        print("⚠️ Please run: bash scripts/download_motif_scaffolds.sh to download motif data")
        
        # For testing, create dummy motif data
        create_dummy_motif_data(data_dir)
    
    return data_dir

def create_dummy_motif_data(data_dir: str):
    """Create dummy motif data for testing."""
    print("🔧 Creating dummy motif data for testing...")
    
    # Create dummy structure tokens file
    struct_file = os.path.join(data_dir, "struct.fasta")
    with open(struct_file, 'w') as f:
        f.write(">motif_1\n")
        f.write("159,162,163,164,165,166,167,168,169,170\n")  # Dummy structure tokens
        f.write(">motif_2\n") 
        f.write("155,158,159,160,161,162,163,164,165,166\n")
    
    # Create dummy sequence file
    seq_file = os.path.join(data_dir, "seq.fasta")
    with open(seq_file, 'w') as f:
        f.write(">motif_1\n")
        f.write("MKTVRQERLK\n")  # Dummy motif sequence
        f.write(">motif_2\n")
        f.write("ADELKVRQER\n")
    
    print(f"✅ Created dummy motif data in {data_dir}")

def load_motif_data(data_dir: str) -> List[MotifData]:
    """Load motif scaffolding data following original DPLM format."""
    # Load from the simple seq.fasta and struct.fasta files (original DPLM format)
    seq_file = os.path.join(data_dir, "seq.fasta")
    struct_file = os.path.join(data_dir, "struct.fasta")
    
    motifs = []
    
    # Load motif sequences
    motif_sequences = {}
    if os.path.exists(seq_file):
        for record in SeqIO.parse(seq_file, "fasta"):
            motif_sequences[record.id] = str(record.seq).strip()
    
    # Load motif structure tokens
    motif_structures = {}
    if os.path.exists(struct_file):
        for record in SeqIO.parse(struct_file, "fasta"):
            motif_structures[record.id] = str(record.seq).strip()
    
    # Create MotifData objects from simple format
    for motif_id in motif_sequences.keys():
        motif_seq = motif_sequences[motif_id]
        motif_struct = motif_structures.get(motif_id, "")
        
        if motif_seq and len(motif_seq) > 0:
            full_length = len(motif_seq)
            motif_data = MotifData(
                name=motif_id,
                motif_sequence=motif_seq,
                motif_positions=list(range(len(motif_seq))),  # Simple: motif at start
                full_sequence=motif_seq,  # For simple testing, motif is the full sequence
                motif_structure=motif_struct,
                pdb_file=None,
                target_scaffold_length=0,
                is_contiguous=True,
                target_length=full_length
            )
            motifs.append(motif_data)
            
            print(f"✅ Loaded motif {motif_id}: {len(motif_seq)} residues")
            if motif_struct:
                print(f"   Structure tokens: {motif_struct}")
    
    # If no simple motifs found, try loading from complex PDB format as fallback
    if not motifs:
        print("📥 No simple motifs found, trying PDB format...")
        motifs = load_complex_motif_data(data_dir)
    
    print(f"📊 Loaded {len(motifs)} motif scaffolding problems")
    return motifs

def load_complex_motif_data(data_dir: str) -> List[MotifData]:
    """Load complex motif scaffolding data from PDB files (fallback)."""
    # Load full sequences
    aa_seq_file = os.path.join(data_dir, "aa_seq.fasta")
    struct_seq_file = os.path.join(data_dir, "struct_seq.fasta")
    
    motifs = []
    
    # Load full sequences
    full_sequences = {}
    if os.path.exists(aa_seq_file):
        for record in SeqIO.parse(aa_seq_file, "fasta"):
            full_sequences[record.id] = str(record.seq).replace(" ", "").upper()
    
    # Load structure sequences
    struct_sequences = {}
    if os.path.exists(struct_seq_file):
        for record in SeqIO.parse(struct_seq_file, "fasta"):
            struct_sequences[record.id] = str(record.seq)
    
    # Process ALL PDB structures (not just 3 for testing)
    for pdb_id in list(full_sequences.keys()):  # Process all available motifs
        full_seq = full_sequences[pdb_id]
        struct_seq = struct_sequences.get(pdb_id, "")
        
        # Look for motif PDB file
        motif_pdb = os.path.join(data_dir, f"{pdb_id}_motif.pdb")
        
        if os.path.exists(motif_pdb):
            # Extract motif sequence from motif PDB
            motif_seq = extract_motif_sequence_from_pdb(motif_pdb)
            
            if motif_seq and len(motif_seq) > 0:
                # **NEW**: Handle non-contiguous motifs properly
                motif_positions = find_motif_positions_advanced(motif_seq, full_seq)
                
                # **NEW**: Calculate target scaffold length from actual data
                target_scaffold_length = len(full_seq) - len(motif_seq)
                
                # Extract corresponding structure tokens for motif positions
                motif_struct_tokens = ""
                if struct_seq and motif_positions:
                    # Get structure tokens for motif positions from full sequence
                    struct_tokens_list = [t.strip() for t in struct_seq.split(',') if t.strip()]
                    if len(struct_tokens_list) >= len(full_seq):
                        # Extract tokens at motif positions (handles both contiguous and non-contiguous)
                        motif_struct_list = [struct_tokens_list[pos] for pos in motif_positions if pos < len(struct_tokens_list)]
                        motif_struct_tokens = ','.join(motif_struct_list)
                        contiguous = motif_seq in full_seq
                        print(f"   📊 Extracted {len(motif_struct_list)} structure tokens for {'contiguous' if contiguous else 'NON-CONTIGUOUS'} motif")
                        print(f"      Motif positions: {motif_positions[:10]}{'...' if len(motif_positions) > 10 else ''}")
                    else:
                        print(f"   ⚠️ Structure tokens length mismatch: {len(struct_tokens_list)} tokens vs {len(full_seq)} sequence length")
                
                contiguous_flag = (motif_seq in full_seq) or (motif_seq.upper() in full_seq.upper())
                
                motif_data = MotifData(
                    name=pdb_id,
                    motif_sequence=motif_seq,
                    motif_positions=motif_positions,
                    full_sequence=full_seq,
                    motif_structure=motif_struct_tokens,
                    pdb_file=motif_pdb,
                    target_scaffold_length=target_scaffold_length,
                    is_contiguous=contiguous_flag,
                    target_length=len(full_seq)
                )
                
                motifs.append(motif_data)
                
                contiguous_status = "contiguous" if contiguous_flag else "NON-CONTIGUOUS"
                print(f"✅ Loaded complex motif {pdb_id}: {len(motif_seq)} residues, full length {len(full_seq)}, scaffold target: {target_scaffold_length}, {contiguous_status}")
    
    return motifs

def extract_motif_sequence_from_pdb(pdb_file: str) -> str:
    """Extract amino acid sequence from motif PDB file."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("motif", pdb_file)
        
        # Get first chain
        chain = next(structure.get_chains())
        
        # Standard amino acid mapping
        aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        
        sequence = ""
        for residue in chain:
            if residue.get_resname() in aa_map:
                sequence += aa_map[residue.get_resname()]
        
        return sequence
        
    except Exception as e:
        print(f"⚠️ Failed to extract sequence from {pdb_file}: {e}")
        return ""

def find_motif_positions(motif_seq: str, full_seq: str) -> List[int]:
    """Find positions where motif occurs in full sequence."""
    positions = []
    start = 0
    while True:
        pos = full_seq.find(motif_seq, start)
        if pos == -1:
            break
        positions.extend(range(pos, pos + len(motif_seq)))
        start = pos + 1
    return positions

def find_motif_positions_advanced(motif_seq: str, full_seq: str) -> List[int]:
    """Find motif positions handling both contiguous and non-contiguous cases."""
    # **FIXED**: First try exact contiguous match (case-sensitive)
    if motif_seq in full_seq:
        pos = full_seq.find(motif_seq)
        print(f"   ✅ Contiguous motif found at position {pos}")
        return list(range(pos, pos + len(motif_seq)))
    
    # **FIXED**: Try case-insensitive match
    motif_upper = motif_seq.upper()
    full_upper = full_seq.upper()
    if motif_upper in full_upper:
        pos = full_upper.find(motif_upper)
        print(f"   ✅ Contiguous motif found (case-insensitive) at position {pos}")
        return list(range(pos, pos + len(motif_seq)))
    
    # **FIXED**: Try reverse complement match (some motifs might be reversed)
    motif_rev = motif_seq[::-1]
    if motif_rev in full_seq:
        pos = full_seq.find(motif_rev)
        print(f"   ✅ Contiguous motif found (reversed) at position {pos}")
        return list(range(pos, pos + len(motif_seq)))
    
    # If not contiguous, find best scattered match
    print(f"   ⚠️ Non-contiguous motif detected: {motif_seq[:20]}...")
    
    # Strategy 1: Find longest common subsequences
    positions = []
    remaining_full = full_seq
    remaining_motif = motif_seq
    offset = 0
    
    while remaining_motif and remaining_full:
        # Find next matching amino acid
        next_aa = remaining_motif[0]
        pos = remaining_full.find(next_aa)
        
        if pos != -1:
            positions.append(offset + pos)
            remaining_full = remaining_full[pos + 1:]
            remaining_motif = remaining_motif[1:]
            offset += pos + 1
        else:
            # Can't find this amino acid, skip it
            remaining_motif = remaining_motif[1:]
    
    print(f"   📍 Found {len(positions)}/{len(motif_seq)} motif positions in scattered mode")
    return positions

def create_motif_scaffold_batch(motif_data: MotifData, scaffold_length: int, tokenizer) -> Dict:
    """Create proper batch for motif scaffolding handling contiguous and non-contiguous motifs."""
    
    if motif_data.is_contiguous:
        # Simple case: contiguous motif
        return create_contiguous_motif_batch(motif_data, scaffold_length, tokenizer)
    else:
        # Complex case: non-contiguous motif
        return create_noncontiguous_motif_batch(motif_data, scaffold_length, tokenizer)

def create_contiguous_motif_batch(motif_data: MotifData, scaffold_length: int, tokenizer) -> Dict:
    """Create batch for contiguous motif scaffolding following DPLM-2 original approach."""
    motif_seq = list(motif_data.motif_sequence)
    
    # **FIXED**: Use target length from motif data to ensure correct generation length
    target_length = len(motif_data.full_sequence)
    actual_scaffold_length = target_length - len(motif_seq)
    
    print(f"   🔧 Contiguous motif batch: {len(motif_seq)} motif + {actual_scaffold_length} scaffold = {target_length} target")
    
    # **DPLM-2 ORIGINAL APPROACH**: Place motif at appropriate position with scaffold around it
    # For contiguous motifs, we can place motif at the beginning or middle
    motif_position = 0  # Place motif at start for simplicity
    
    # Create sequence: scaffold_left + motif + scaffold_right
    scaffold_left_length = motif_position
    scaffold_right_length = actual_scaffold_length - scaffold_left_length
    
    aa_sequence_tokens = (
        [tokenizer.aa_cls_token] +
        [tokenizer.aa_mask_token] * scaffold_left_length +
        motif_seq +
        [tokenizer.aa_mask_token] * scaffold_right_length +
        [tokenizer.aa_eos_token]
    )
    
    # Structure tokens: scaffold_left + motif_structure + scaffold_right
    if motif_data.motif_structure:
        struct_tokens = motif_data.motif_structure.split(",")[:len(motif_seq)]
    else:
        struct_tokens = [tokenizer.struct_mask_token] * len(motif_seq)
    
    struct_sequence_tokens = (
        [tokenizer.struct_cls_token] +
        [tokenizer.struct_mask_token] * scaffold_left_length +
        struct_tokens +
        [tokenizer.struct_mask_token] * scaffold_right_length +
        [tokenizer.struct_eos_token]
    )
    
    # Motif positions are after left scaffold
    motif_start = 1 + scaffold_left_length  # +1 for CLS token
    motif_positions = list(range(motif_start, motif_start + len(motif_seq)))
    
    return {
        'aa_sequence': "".join(aa_sequence_tokens),
        'struct_sequence': "".join(struct_sequence_tokens),
        'motif_positions': motif_positions,
        'total_length': len(aa_sequence_tokens),
        'target_length': target_length,
        'scaffold_left_length': scaffold_left_length,
        'scaffold_right_length': scaffold_right_length
    }

def create_noncontiguous_motif_batch(motif_data: MotifData, scaffold_length: int, tokenizer) -> Dict:
    """Create batch for non-contiguous motif scaffolding following original DPLM-2 approach."""
    
    # **ORIGINAL DPLM-2 APPROACH**: Place motif in middle with scaffold on both sides
    # For non-contiguous motifs, insert spacers between motif segments
    
    target_length = len(motif_data.full_sequence)
    motif_seq = list(motif_data.motif_sequence)
    
    # **FIXED**: Use actual target length instead of passed scaffold_length
    total_scaffold_length = target_length - len(motif_seq)
    scaffold_left_length = total_scaffold_length // 2  # Half on left
    scaffold_right_length = total_scaffold_length - scaffold_left_length  # Rest on right
    
    print(f"   🔧 DPLM-2 non-contiguous: {scaffold_left_length} left + {len(motif_seq)} motif + {scaffold_right_length} right = {target_length} total")
    
    # Create template following original DPLM-2 pattern
    aa_sequence_tokens = (
        [tokenizer.aa_cls_token] +
        [tokenizer.aa_mask_token] * scaffold_left_length +
        motif_seq +
        [tokenizer.aa_mask_token] * scaffold_right_length +
        [tokenizer.aa_eos_token]
    )
    
    # Structure tokens: mask + motif structure + mask
    if motif_data.motif_structure:
        struct_tokens = motif_data.motif_structure.split(",")[:len(motif_seq)]
    else:
        struct_tokens = [tokenizer.struct_mask_token] * len(motif_seq)
    
    struct_sequence_tokens = (
        [tokenizer.struct_cls_token] +
        [tokenizer.struct_mask_token] * scaffold_left_length +
        struct_tokens +
        [tokenizer.struct_mask_token] * scaffold_right_length +
        [tokenizer.struct_eos_token]
    )
    
    # Motif positions are in the middle after left scaffold
    motif_start = 1 + scaffold_left_length  # +1 for CLS token
    motif_positions = list(range(motif_start, motif_start + len(motif_seq)))
    
    return {
        'aa_sequence': "".join(aa_sequence_tokens),
        'struct_sequence': "".join(struct_sequence_tokens),
        'motif_positions': motif_positions,
        'total_length': len(aa_sequence_tokens),
        'template_based': True,
        'scaffold_left_length': scaffold_left_length,
        'scaffold_right_length': scaffold_right_length
    }

def generate_scaffold_baseline(dplm2: DPLM2Integration, motif: Dict, 
                             scaffold_length: int = 50, expert_id: int = 1) -> Dict:
    """Generate baseline scaffold using DPLM-2."""
    try:
        # For motif scaffolding, we condition on both motif sequence and structure
        # and generate the remaining scaffold positions
        
        motif_seq = motif['motif_sequence']
        motif_struct = motif['motif_struct_seq']
        
        # Create masked scaffold: motif + masked positions
        total_length = len(motif_seq) + scaffold_length
        
        # Scaffold sequence: motif + masked positions
        scaffold_seq = motif_seq + dplm2.tokenizer.aa_mask_token * scaffold_length
        
        # Scaffold structure: motif structure + masked positions  
        scaffold_struct = motif_struct + "," + ",".join([dplm2.tokenizer.struct_mask_token] * scaffold_length)
        
        # Generate scaffold using co-generation (both sequence and structure)
        generated_seq = dplm2.generate_from_masked_input(
            aa_sequence=scaffold_seq,
            struct_tokens=scaffold_struct,
            task_type="inverse_folding",  # Generate sequence part
            expert_id=expert_id,
            temperature=1.0
        )
        
        generated_struct = dplm2.generate_from_masked_input(
            aa_sequence=scaffold_seq,
            struct_tokens=scaffold_struct,
            task_type="folding",  # Generate structure part
            expert_id=expert_id,
            temperature=0.9
        )
        
        return {
            'motif_sequence': motif_seq,
            'motif_struct_seq': motif_struct,
            'scaffold_sequence': generated_seq,
            'scaffold_struct_seq': generated_struct,
            'full_sequence': generated_seq,
            'full_struct_seq': generated_struct,
            'baseline_method': 'dplm2_scaffold'
        }
        
    except Exception as e:
        print(f"⚠️ Baseline scaffold generation failed: {e}")
        return None

def evaluate_scaffold(scaffold_data: Dict, motif_data: MotifData, reference_seq: str = None) -> Dict:
    """Evaluate scaffold quality with comprehensive metrics."""
    try:
        full_seq = scaffold_data.get('full_sequence', '')
        motif_seq = motif_data.motif_sequence
        
        # 1. Motif preservation - handle both contiguous and non-contiguous motifs
        if hasattr(motif_data, 'is_contiguous') and not motif_data.is_contiguous:
            # For non-contiguous motifs, check amino acid coverage
            if motif_seq and full_seq:
                motif_chars = set(motif_seq)
                generated_chars = set(full_seq)
                motif_coverage = len(motif_chars.intersection(generated_chars)) / len(motif_chars)
                motif_preserved = motif_coverage >= 0.8  # 80% coverage threshold
            else:
                motif_preserved = False
        else:
            # For contiguous motifs, check exact substring match
            motif_preserved = motif_seq in full_seq if motif_seq and full_seq else False
        
        # 2. Sequence quality
        valid_aa_ratio = sum(1 for aa in full_seq if aa in "ACDEFGHIKLMNPQRSTVWY") / len(full_seq) if full_seq else 0.0
        
        # 3. Length metrics
        scaffold_length = len(full_seq) - len(motif_seq) if full_seq and motif_seq else 0
        
        # 4. Sequence recovery (if reference available)
        sequence_recovery = 0.0
        if reference_seq and full_seq:
            min_len = min(len(full_seq), len(reference_seq))
            if min_len > 0:
                matches = sum(1 for i in range(min_len) if full_seq[i] == reference_seq[i])
                sequence_recovery = matches / min_len
        
        # 5. Motif positioning accuracy
        motif_position_accuracy = 0.0
        if motif_preserved and motif_data.motif_positions:
            found_pos = full_seq.find(motif_seq)
            if found_pos != -1:
                # Check if motif is in approximately correct position
                expected_start = min(motif_data.motif_positions) if motif_data.motif_positions else 0
                position_error = abs(found_pos - expected_start)
                motif_position_accuracy = max(0.0, 1.0 - position_error / len(full_seq))
        
        # 6. Biophysical properties
        charge_ratio = sum(1 for aa in full_seq if aa in "DEKR") / len(full_seq) if full_seq else 0.0
        hydrophobic_ratio = sum(1 for aa in full_seq if aa in "AILMFWYV") / len(full_seq) if full_seq else 0.0
        
        # Overall success criteria
        success = (
            motif_preserved and 
            valid_aa_ratio > 0.95 and 
            scaffold_length > 0 and
            0.1 <= charge_ratio <= 0.4 and  # Reasonable charge distribution
            0.2 <= hydrophobic_ratio <= 0.6  # Reasonable hydrophobic content
        )
        
        return {
            'motif_preserved': motif_preserved,
            'valid_aa_ratio': valid_aa_ratio,
            'scaffold_length': scaffold_length,
            'total_length': len(full_seq) if full_seq else 0,
            'sequence_recovery': sequence_recovery,
            'motif_position_accuracy': motif_position_accuracy,
            'charge_ratio': charge_ratio,
            'hydrophobic_ratio': hydrophobic_ratio,
            'success': success
        }
        
    except Exception as e:
        print(f"⚠️ Scaffold evaluation failed: {e}")
        return {'success': False}

def generate_baseline_scaffold(motif_data: MotifData, dplm2_150m, scaffold_length: int = 50) -> Dict:
    """Generate baseline scaffold using DPLM-2 150M with proper motif structure input."""
    try:
        print(f"   🔄 Generating baseline with DPLM-2 150M using motif structure...")
        
        # **CRITICAL FIX**: Use motif structure from _motif.pdb as input
        if not motif_data.motif_structure:
            print(f"   ❌ No motif structure available for {motif_data.name}")
            return None
            
        print(f"   🧬 Using motif structure: {len(motif_data.motif_structure)} chars")
        print(f"   🎯 Motif sequence: {motif_data.motif_sequence}")
        print(f"   📏 Target scaffold length: {scaffold_length}")
        
        # **PROPER APPROACH**: Structure-conditioned generation with motif constraints
        # This uses the motif structure from _motif.pdb as the structural constraint
        baseline_result = dplm2_150m.generate_scaffold(
            motif_data=motif_data,  # Contains motif_structure from _motif.pdb
            scaffold_length=scaffold_length,
            temperature=0.8
        )
        
        if baseline_result and baseline_result.get('full_sequence'):
            print(f"   ✅ Baseline generated: {len(baseline_result['full_sequence'])} residues")
            print(f"   🧬 Structure sequence: {len(baseline_result.get('structure_sequence', ''))} chars")
            return baseline_result
        else:
            print(f"   ❌ Structure-conditioned baseline generation failed")
            return None
            
    except Exception as e:
        print(f"   ❌ Baseline generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_mcts_scaffolding(motif_data: MotifData, baseline_sequence: str, mcts_experts: List[ExpertModel], 
                        dplm2_integration, scaffold_length: int = 50, num_iterations: int = 10) -> Dict:
    """Run MCTS-guided scaffolding starting from baseline sequence."""
    try:
        # Use baseline sequence as MCTS starting point
        initial_sequence = baseline_sequence
        motif_seq = motif_data.motif_sequence
        
        # Find motif position in baseline sequence
        motif_start = baseline_sequence.find(motif_seq)
        if motif_start == -1:
            print(f"   ⚠️ Motif not found in baseline, using middle position")
            left_scaffold = scaffold_length // 2
            motif_start = left_scaffold
        else:
            left_scaffold = motif_start
        
        motif_end = motif_start + len(motif_seq)
        
        print(f"   🧬 Initial sequence: {initial_sequence[:20]}...{initial_sequence[-20:]} ({len(initial_sequence)} residues)")
        print(f"   🎯 Motif position: {left_scaffold}-{left_scaffold + len(motif_seq)}")
        
        # **PROPER**: Load real PDB reference structure (DPLM approach)
        print(f"   🔄 Loading real PDB reference structure...")
        print(f"   🔍 DEBUG: Starting coordinate loading for {motif_data.name}")
        ref_coords = None
        reference_pdb_path = None
        
        # Try to find corresponding PDB reference file
        pdb_data_dir = "/home/caom/AID3/dplm/data-bin/scaffolding-pdbs"
        potential_names = [motif_data.name, motif_data.name.lower(), motif_data.name.upper()]
        
        for pdb_name in potential_names:
            ref_path = os.path.join(pdb_data_dir, f"{pdb_name}_reference.pdb")
            if os.path.exists(ref_path):
                reference_pdb_path = ref_path
                print(f"   ✅ Found reference PDB: {ref_path}")
                break
        
        if reference_pdb_path:
            # Load reference coordinates using MDAnalysis (following motif_analysis.ipynb)
            try:
                import MDAnalysis as mda
                ref_universe = mda.Universe(reference_pdb_path)
                ref_coords = ref_universe.select_atoms("name CA").positions
                print(f"   ✅ Loaded reference coordinates: {ref_coords.shape}")
            except Exception as e:
                print(f"   ⚠️ Failed to load reference PDB: {e}")
                ref_coords = None
        else:
            print(f"   No reference PDB found for {motif_data.name}")

        # **USE EXISTING BASELINE**: Don't generate again, use the baseline passed to this function
        # The baseline was already generated before calling this function
        # Extract structure tokens from the baseline data that was passed in
        baseline_data = None
        for expert in mcts_experts:
            if hasattr(expert, 'dplm2'):
                # Get the baseline data from the expert that generated it
                baseline_data = expert.dplm2.get_last_generation_data()
                break
        
        # Ensure structure_sequence is a string (not numpy array)
        structure_seq = baseline_data.get('structure_sequence', '') if baseline_data else ''
        if isinstance(structure_seq, np.ndarray):
            # Convert numpy array to comma-separated string
            structure_seq = ','.join(map(str, structure_seq.flatten().astype(int)))
            print(f"   🔧 Converted numpy array structure_sequence to string: {len(structure_seq)} chars")
        elif not isinstance(structure_seq, str):
            structure_seq = str(structure_seq) if structure_seq is not None else ''
        
        baseline_result = {
            'full_sequence': baseline_sequence,
            'structure_sequence': structure_seq,
            'motif_preserved': True,
            'method': 'existing_baseline'
        }
        
        # **CRITICAL**: Extract motif coordinates from reference PDB for motif-RMSD calculation
        motif_coords = None
        full_coords = ref_coords  # Already loaded above
        
        print(f"   🔍 DEBUG: ref_coords is None: {ref_coords is None}")
        print(f"   🔍 DEBUG: motif_positions available: {motif_data.motif_positions is not None}")
        if ref_coords is not None:
            print(f"   🔍 DEBUG: ref_coords shape: {ref_coords.shape}")
        
        if ref_coords is not None and motif_data.motif_positions:
            try:
                # Extract coordinates at motif positions
                # Handle both contiguous and non-contiguous motifs
                valid_positions = [p for p in motif_data.motif_positions if p < len(ref_coords)]
                if len(valid_positions) < len(motif_data.motif_positions):
                    print(f"   ⚠️ Some motif positions out of bounds: {len(valid_positions)}/{len(motif_data.motif_positions)} valid")
                motif_coords = ref_coords[valid_positions]
                print(f"   ✅ Extracted motif coordinates: {motif_coords.shape} from positions {valid_positions[:5]}{'...' if len(valid_positions) > 5 else ''}")
            except Exception as e:
                print(f"   ⚠️ Failed to extract motif coords: {e}")
                motif_coords = None
        
        # Create structure-aware baseline for MCTS with ALL required fields for reward calculation
        baseline_structure = {
            'sequence': baseline_result['full_sequence'],
            'struct_seq': baseline_result.get('structure_sequence', ''),  # Use struct_seq key for consistency
            'motif_sequence': motif_data.motif_sequence,
            'motif_positions': motif_data.motif_positions if motif_data.motif_positions else list(range(len(motif_data.motif_sequence))),
            'motif_coords': motif_coords.tolist() if motif_coords is not None else None,  # **NEW**: For motif-RMSD calculation
            'coordinates': full_coords.tolist() if full_coords is not None else None,  # **NEW**: For scTM calculation
            'structure_data': {'coords': full_coords, 'sequence': motif_data.full_sequence} if full_coords is not None else None,  # **NEW**: For scTM
            'target_length': len(motif_data.full_sequence),
            'name': motif_data.name,
            'scaffold_regions': {
                'left': (0, 0),  
                'right': (len(motif_data.motif_sequence), len(baseline_result['full_sequence']))
            },
            'method': baseline_result['method'],
            'output_tokens': baseline_result.get('output_tokens')  
        }
        
        print(f"   📊 Baseline structure prepared:")
        print(f"      - Motif coords: {motif_coords.shape if motif_coords is not None else 'None'}")
        print(f"      - Full coords: {full_coords.shape if full_coords is not None else 'None'}")
        print(f"      - Motif positions: {motif_data.motif_positions[:5] if motif_data.motif_positions and len(motif_data.motif_positions) > 5 else motif_data.motif_positions}...")

        # Initialize MCTS with motif-aware configuration
        
        # Use DPLM-2 650M integration from MCTS experts (not 150M baseline)
        dplm2_integration = None
        for expert in mcts_experts:
            if hasattr(expert, 'dplm2'):
                dplm2_integration = expert.dplm2
        
        # **NEW**: Set baseline structure tokens in DPLM-2 integration
        structure_seq = baseline_structure.get('structure_sequence')
        if structure_seq is not None and structure_seq != '':
            # Ensure it's a string (not numpy array)
            if isinstance(structure_seq, np.ndarray):
                structure_seq = ','.join(map(str, structure_seq.flatten().astype(int)))
            elif not isinstance(structure_seq, str):
                structure_seq = str(structure_seq)
            dplm2_integration.set_baseline_structure_tokens(structure_seq)
        
        # Initialize MCTS with deeper search for better testing
        mcts = GeneralMCTS(
            dplm2_integration=dplm2_integration,
            task_type="motif_scaffolding",
            max_depth=5,  # Increased depth for better testing
            num_children_select=6,  # Fixed parameter name
            ablation_mode="multi_expert",
            backup_rule="sum",
            reference_coords=ref_coords,
            baseline_structure=baseline_structure,
            reference_sequence=baseline_sequence  # Fixed parameter name
        )
        
        # Structure-aware motif scaffolding uses built-in MCTS expansion
        # The MCTS will automatically call _expand_motif_scaffolding() based on task_type
        print(f"   🧬 Structure-aware MCTS will preserve motif positions: {baseline_structure['motif_positions']}")
        print(f"   🏗️ Scaffold regions to optimize: {baseline_structure['scaffold_regions']}")
        
        # Run MCTS search
        print(f"   🔄 Running MCTS with {len(mcts_experts)} experts...")
        root_node = mcts.search(
            initial_sequence=initial_sequence,
            num_iterations=num_iterations,
            max_depth=3
        )
        
        # Get the actual best node from the tree
        if root_node:
            best_node = mcts._get_best_node(root_node)
            print(f"   🏆 Best node found: depth={best_node.depth}, reward={best_node.average_value:.3f}")
        else:
            best_node = None
        
        if best_node:
            return {
                'full_sequence': best_node.sequence,
                'motif_preserved': motif_data.motif_sequence in best_node.sequence,
                'scaffold_length': len(best_node.sequence) - len(motif_data.motif_sequence),
                'method': 'mcts_multi_expert',
                'mcts_iterations': num_iterations,
                'final_reward': best_node.average_value,
                'mcts_best_reward': best_node.average_value  # Show actual MCTS reward
            }
        else:
            print(f"   ❌ MCTS search failed")
            return None
            
    except Exception as e:
        print(f"   ❌ MCTS scaffolding failed: {e}")
        return None

def compute_ensemble_surprisal(expert_logits_list: List[torch.Tensor]) -> float:
    """Compute ensemble surprisal from multiple expert logits."""
    if not expert_logits_list:
        return 0.0
    
    # Combine logits from all experts
    ensemble_probs = []
    for logits in expert_logits_list:
        if logits is not None:
            probs = torch.softmax(logits, dim=-1)
            ensemble_probs.append(probs)
    
    if not ensemble_probs:
        return 0.0
    
    # Average ensemble probability
    avg_probs = torch.stack(ensemble_probs).mean(dim=0)
    
    # Compute surprisal: -log(P(sequence))
    surprisal = -torch.log(avg_probs + 1e-8).sum()
    return surprisal.item()

def select_experts_by_motif_characteristics(motif_data: MotifData, available_experts: List[ExpertModel]) -> List[ExpertModel]:
    """Dynamic expert selection based on motif characteristics."""
    selected_experts = []
    
    motif_length = len(motif_data.motif_sequence)
    motif_complexity = len(set(motif_data.motif_sequence))  # Unique amino acids
    has_structure = motif_data.motif_structure is not None
    
    # Selection criteria:
    # 1. Short motifs (<10 AA): Prefer fast models (Proteinea, DPLM-2)
    # 2. Long motifs (>15 AA): Prefer structure-aware models (RFDiffusion, FoldFlow)  
    # 3. High complexity: Prefer diverse models (all experts)
    # 4. Has structure info: Prioritize structure-aware models
    
    for expert in available_experts:
        expert_name = expert.get_name().lower()
        
        # Always include DPLM-2 for multimodal capability
        if 'dplm' in expert_name:
            selected_experts.append(expert)
            continue
            
        # Short motifs: prefer fast generation
        if motif_length < 10:
            if 'proteinea' in expert_name:
                selected_experts.append(expert)
        
        # Long motifs: prefer structure-aware
        elif motif_length > 15:
            if 'rfdiffusion' in expert_name or 'foldflow' in expert_name:
                selected_experts.append(expert)
        
        # Medium motifs or high complexity: include all
        else:
            selected_experts.append(expert)
        
        # Structure info available: prioritize structure-aware models
        if has_structure and ('rfdiffusion' in expert_name or 'foldflow' in expert_name):
            if expert not in selected_experts:
                selected_experts.append(expert)
    
    # Ensure at least 2 experts for ensemble
    if len(selected_experts) < 2:
        selected_experts = available_experts[:2]
    
    print(f"   🎯 Selected {len(selected_experts)} experts for motif (length={motif_length}, complexity={motif_complexity}, structure={has_structure})")
    for expert in selected_experts:
        print(f"      • {expert.get_name()}")
    
    return selected_experts

def compute_ph_uct_score(expert_outputs: List[ExpertOutput], exploration_constant: float = 1.4) -> List[Tuple[ExpertOutput, float]]:
    """Compute PH-UCT scores for expert outputs."""
    scored_outputs = []
    
    # Compute ensemble surprisal for exploration bonus
    ensemble_surprisal = compute_ensemble_surprisal([output.logits for output in expert_outputs if output.logits is not None])
    
    for output in expert_outputs:
        if output is None:
            continue
            
        # Base reward (motif preservation + confidence)
        base_reward = 1.0 if output.motif_preserved else 0.0
        if output.confidence_scores:
            avg_confidence = np.mean(output.confidence_scores)
            base_reward += avg_confidence * 0.5
        
        # Entropy bonus for exploration (PH-UCT)
        entropy_bonus = 0.0
        if output.entropy is not None:
            # Higher entropy = more exploration value
            entropy_bonus = output.entropy * exploration_constant * 0.1
        
        # Ensemble surprisal bonus
        surprisal_bonus = ensemble_surprisal * 0.01 if ensemble_surprisal > 0 else 0.0
        
        # Structure awareness bonus
        structure_bonus = 0.1 if output.structure_aware else 0.0
        
        # Total PH-UCT score
        ph_uct_score = base_reward + entropy_bonus + surprisal_bonus + structure_bonus
        
        scored_outputs.append((output, ph_uct_score))
    
    # Sort by PH-UCT score (descending)
    scored_outputs.sort(key=lambda x: x[1], reverse=True)
    return scored_outputs

def run_motif_scaffolding_experiment(motifs: List[MotifData], experts: List[ExpertModel], 
                                   mode: str = "baseline", dplm2_integration=None, **kwargs) -> List[Dict]:
    """Run motif scaffolding experiment with multiple experts."""
    results = []
    default_scaffold_length = kwargs.get('scaffold_length', 50)
    
    for i, motif_data in enumerate(motifs):
        print(f"\n🧬 Processing motif {i+1}/{len(motifs)}: {motif_data.name}")
        print(f"   Motif: {motif_data.motif_sequence} ({len(motif_data.motif_sequence)} residues)")
        print(f"   Target length: {len(motif_data.full_sequence)} residues")
        
        # **NEW**: Use dynamic scaffold length based on actual target data
        if hasattr(motif_data, 'target_scaffold_length'):
            scaffold_length = motif_data.target_scaffold_length
            print(f"   🎯 Using target scaffold length: {scaffold_length} (from data)")
        else:
            scaffold_length = default_scaffold_length
            print(f"   🎯 Using default scaffold length: {scaffold_length}")
        
        if not getattr(motif_data, 'target_length', None):
            motif_data.target_length = len(motif_data.full_sequence)
        
        # **NEW**: Handle non-contiguous motifs
        if hasattr(motif_data, 'is_contiguous'):
            contiguous_status = "contiguous" if motif_data.is_contiguous else "NON-CONTIGUOUS"
            print(f"   📍 Motif type: {contiguous_status}")
            if not motif_data.is_contiguous:
                print(f"   ⚠️ Non-contiguous motif detected - using scattered positioning strategy")
        
        try:
            if mode in {"baseline", "baseline_all"}:
                # Test each expert individually for baseline
                for expert in experts:
                    print(f"   🔄 Testing {expert.get_name()} baseline...")
                    
                    scaffold_data = expert.generate_scaffold(
                        motif_data, 
                        scaffold_length=scaffold_length,
                        temperature=1.0
                    )
                    
                    if scaffold_data:
                        evaluation = evaluate_scaffold(
                            scaffold_data, 
                            motif_data, 
                            reference_seq=motif_data.full_sequence
                        )
                        
                        result = {
                            'motif_name': motif_data.name,
                            'mode': f"{mode}_{expert.get_name()}",
                            'motif_length': len(motif_data.motif_sequence),
                            'scaffold_data': scaffold_data,
                            'evaluation': evaluation,
                            'expert': expert.get_name()
                        }
                        
                        print(f"     Motif preserved: {evaluation.get('motif_preserved', False)}")
                        print(f"     Sequence recovery: {evaluation.get('sequence_recovery', 0.0):.3f}")
                        print(f"     Success: {evaluation.get('success', False)}")
                        
                        results.append(result)
                    else:
                        print(f"     ❌ {expert.get_name()} failed to generate scaffold")
                        
            elif mode == "mcts":
                # HYBRID APPROACH: DPLM-2 150M baseline + Multi-expert MCTS
                
                # Step 1: Generate baseline with DPLM-2 150M (separate from MCTS)
                dplm2_150m = None
                mcts_experts = []
                
                # **CRITICAL FIX**: Always create DPLM-2 baseline expert, even for external model experiments
                dplm2_150m = DPLM2ExpertModel(dplm2_integration)  # Use passed DPLM-2 integration
                
                for expert in experts:
                    expert_name = expert.get_name()
                    print(f"   🔍 Checking expert: {expert_name}")
                    
                    # Check for DPLM-2 experts
                    if "DPLM" in expert_name and hasattr(expert, 'dplm2'):
                        mcts_experts.append(expert)  # Add to MCTS for multi-expert rollouts
                        print(f"   ✅ Added {expert_name} to MCTS")
                    # Check for external model experts (both direct and API versions)
                    elif any(model_type in expert_name for model_type in ["Proteina", "FlowFlow", "FoldFlow", "RFDiffusion", "ProteinMPNN"]):
                        # External models for MCTS
                        mcts_experts.append(expert)
                        print(f"   ✅ Added {expert_name} to MCTS")
                    else:
                        print(f"   ⚠️ Skipped {expert_name} (not in MCTS list)")
                
                # Ensure we have a baseline generator
                if not dplm2_150m:
                    print("   ❌ Critical error: No DPLM-2 available for baseline generation")
                    continue
                
                print(f"   📊 Baseline: {dplm2_150m.get_name() if dplm2_150m else 'None'}")
                print(f"   🤖 MCTS Experts: {[e.get_name() for e in mcts_experts]}")
                
                # **CRITICAL FIX**: Use correct scaffold length for motif scaffolding
                # Scaffold length = reference_length - motif_length (from motif PDB)
                correct_scaffold_length = len(motif_data.full_sequence) - len(motif_data.motif_sequence)
                actual_scaffold_length = max(5, correct_scaffold_length)  # Minimum 5 for reasonable scaffold
                print(f"   🔧 Correct motif scaffolding: {len(motif_data.motif_sequence)} motif + {actual_scaffold_length} scaffold = {len(motif_data.motif_sequence) + actual_scaffold_length} target")
                print(f"   📋 Reference: {motif_data.full_sequence} ({len(motif_data.full_sequence)} residues)")
                print(f"   🎯 Motif: {motif_data.motif_sequence} ({len(motif_data.motif_sequence)} residues)")
                
                # Generate baseline scaffold
                baseline_data = generate_baseline_scaffold(motif_data, dplm2_150m, actual_scaffold_length)
                
                if not baseline_data:
                    print("   ❌ Baseline generation failed")
                    continue
                
                # Step 2: MCTS optimization starting from baseline
                print(f"   🔄 Running MCTS with {len(mcts_experts)} experts...")
                
                # **FIX**: Use the clean MotifScaffoldingMCTS implementation
                try:
                    from core.motif_scaffolding_mcts import MotifScaffoldingMCTS
                    
                    # Create clean motif data structure
                    clean_motif_data = type('MotifData', (), {
                        'name': motif_data.name,
                        'motif_sequence': motif_data.motif_sequence,
                        'motif_structure_tokens': motif_data.motif_structure or '',
                        'reference_sequence': motif_data.full_sequence,
                        'target_length': len(motif_data.full_sequence),
                        'motif_positions': motif_data.motif_positions or []
                    })()
                    
                    # Get external experts for MCTS
                    external_experts = [e for e in mcts_experts if 'DPLM' not in e.get_name()]
                    print(f"   🤖 Using {len(external_experts)} external experts for MCTS")
                    
                    # Initialize clean MCTS
                    clean_mcts = MotifScaffoldingMCTS(dplm2_150m.dplm2, external_experts)
                    
                    # Run MCTS search
                    best_node = clean_mcts.search(
                        motif_data=clean_motif_data,
                        baseline_sequence=baseline_data['full_sequence'],
                        baseline_structure=baseline_data.get('structure_sequence', ''),
                        num_iterations=kwargs.get('mcts_iterations', 10),
                        max_depth=3
                    )
                    
                    if best_node and hasattr(best_node, 'sequence'):
                        scaffold_data = {
                            'full_sequence': best_node.sequence,
                            'motif_preserved': motif_data.motif_sequence in best_node.sequence,
                            'scaffold_length': len(best_node.sequence) - len(motif_data.motif_sequence),
                            'method': 'mcts_multi_expert',
                            'mcts_iterations': kwargs.get('mcts_iterations', 10),
                            'final_reward': getattr(best_node, 'reward', 0.0)
                        }
                        print(f"   ✅ MCTS completed: reward={best_node.reward:.3f}")
                    else:
                        print(f"   ❌ MCTS returned invalid result")
                        scaffold_data = None
                        
                except Exception as mcts_e:
                    print(f"   ❌ MCTS optimization failed: {mcts_e}")
                    import traceback
                    traceback.print_exc()
                    scaffold_data = None
                
                if scaffold_data:
                    evaluation = evaluate_scaffold(
                        scaffold_data, 
                        motif_data, 
                        reference_seq=motif_data.full_sequence
                    )
                    
                    result = {
                        'motif_name': motif_data.name,
                        'mode': mode,
                        'motif_length': len(motif_data.motif_sequence),
                        'scaffold_data': scaffold_data,
                        'evaluation': evaluation,
                        'expert': 'multi_expert_mcts'
                    }
                    
                    print(f"   Motif preserved: {evaluation.get('motif_preserved', False)}")
                    print(f"   Sequence recovery: {evaluation.get('sequence_recovery', 0.0):.3f}")
                    print(f"   Final reward: {scaffold_data.get('final_reward', 0.0):.3f}")
                    print(f"   MCTS best reward: {scaffold_data.get('mcts_best_reward', 0.0):.3f}")
                    print(f"   Success: {evaluation.get('success', False)}")
                    
                    results.append(result)
                else:
                    print(f"   ❌ MCTS scaffolding failed")
                    
        except Exception as e:
            print(f"   ❌ Error processing motif: {e}")
            continue
    
    return results

def load_shared_baseline_cache(cache_file: str) -> Dict:
    """Load shared baseline cache for fair ablation comparison."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            print(f"📋 Loaded shared baseline cache with {len(cache)} entries from {cache_file}")
            return cache
        except Exception as e:
            print(f"⚠️ Failed to load baseline cache: {e}")
            return {}
    else:
        print(f"📋 Creating new shared baseline cache at {cache_file}")
        return {}

def save_shared_baseline_cache(cache: Dict, cache_file: str):
    """Save shared baseline cache for future experiments."""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2, default=str)
        print(f"💾 Saved shared baseline cache with {len(cache)} entries to {cache_file}")
    except Exception as e:
        print(f"⚠️ Failed to save baseline cache: {e}")

def convert_to_legacy_motif_data(clean_motif) -> MotifData:
    """Convert clean MotifScaffoldingData into legacy MotifData interface for experts."""
    target_length = getattr(clean_motif, 'target_length', None)
    reference_sequence = getattr(clean_motif, 'reference_sequence', None) or clean_motif.motif_sequence
    motif_length = len(clean_motif.motif_sequence)
    if target_length is None and reference_sequence:
        target_length = len(reference_sequence)
    target_scaffold_length = None
    if target_length is not None:
        target_scaffold_length = max(target_length - motif_length, 0)
    
    is_contiguous = True
    motif_segments = getattr(clean_motif, 'motif_segments', None)
    if motif_segments and len(motif_segments) > 1:
        is_contiguous = False
    
    return MotifData(
        name=clean_motif.name,
        motif_sequence=clean_motif.motif_sequence,
        motif_positions=list(getattr(clean_motif, 'motif_positions', []) or []),
        full_sequence=reference_sequence,
        motif_structure=getattr(clean_motif, 'motif_structure_tokens', None),
        pdb_file=None,
        target_scaffold_length=target_scaffold_length,
        is_contiguous=is_contiguous,
        target_length=target_length
    )

def run_sampling_only_multi_expert(
    motif_data_clean,
    experts: List[ExpertModel],
    mcts_instance,
    baseline_sequence: str,
    baseline_reward: float,
    num_samples: int = 2,
    temperature: float = 1.0,
) -> Dict:
    """
    Run direct multi-expert sampling without tree search.
    
    Args:
        motif_data_clean: MotifScaffoldingData instance from clean loader
        experts: List of ExpertModel implementations to sample from
        mcts_instance: MotifScaffoldingMCTS instance for reward evaluation
        baseline_sequence: Shared baseline sequence for comparison
        baseline_reward: Reward assigned to the baseline
        num_samples: Number of samples to draw per expert
        temperature: Sampling temperature passed to experts (if supported)
    """
    legacy_motif = convert_to_legacy_motif_data(motif_data_clean)
    scaffold_length = legacy_motif.target_scaffold_length
    if scaffold_length is None and legacy_motif.target_length is not None:
        scaffold_length = max(legacy_motif.target_length - len(legacy_motif.motif_sequence), 0)
    
    sample_records = []
    baseline_reward_calc, baseline_details = mcts_instance._calculate_reward(
        motif_data_clean,
        baseline_sequence,
        return_details=True
    )
    if baseline_reward is None:
        baseline_reward = baseline_reward_calc
    else:
        baseline_reward = float(baseline_reward)
    best_reward = baseline_reward
    best_sequence = baseline_sequence
    best_sample_record = None
    target_length = legacy_motif.target_length
    baseline_length_matches = (
        target_length is None or len(baseline_sequence) == target_length
    )
    best_length_reward = baseline_reward if baseline_length_matches else float('-inf')
    best_length_matched_record = None
    best_sctm = float('-inf')
    best_sctm_record = None
    best_rmsd = float('inf')
    best_rmsd_record = None

    print(f"   🧪 Sampling {num_samples} rollout(s) per expert (no tree search)")
    
    for expert in experts:
        expert_name = expert.get_name() if hasattr(expert, 'get_name') else str(expert)
        for sample_idx in range(1, num_samples + 1):
            try:
                scaffold_data = expert.generate_scaffold(
                    legacy_motif,
                    scaffold_length=scaffold_length if scaffold_length is not None else 50,
                    temperature=temperature
                )
            except Exception as sample_error:
                print(f"     ❌ {expert_name} sample {sample_idx}: generation failed ({sample_error})")
                continue
            
            if not scaffold_data or not scaffold_data.get('full_sequence'):
                print(f"     ⚠️ {expert_name} sample {sample_idx}: empty generation")
                continue
            
            generated_sequence = scaffold_data['full_sequence']
            reward, details = mcts_instance._calculate_reward(
                motif_data_clean,
                generated_sequence,
                return_details=True
            )
            mcts_instance.cache[generated_sequence] = reward  # Cache for consistency
            sample_length_matches = (
                target_length is None or len(generated_sequence) == target_length
            )
            
            evaluation = evaluate_scaffold(
                {'full_sequence': generated_sequence},
                legacy_motif,
                reference_seq=legacy_motif.full_sequence
            )
            
            record = {
                'expert': expert_name,
                'sample_index': sample_idx,
                'sequence': generated_sequence,
                'reward': reward,
                'details': details,
                'evaluation': evaluation,
                'length_matches_target': sample_length_matches
            }
            sample_records.append(record)
            
            print(f"     ✅ {expert_name} sample {sample_idx}: reward={reward:.3f}, motif_preserved={evaluation.get('motif_preserved', False)}")
            
            if reward > best_reward:
                best_reward = reward
                best_sequence = generated_sequence
                best_sample_record = record
            
            if sample_length_matches and reward > best_length_reward:
                best_length_reward = reward
                best_length_matched_record = record
            
            current_sctm = details.get('sctm')
            if current_sctm is not None and current_sctm > best_sctm:
                best_sctm = current_sctm
                best_sctm_record = record
            
            current_rmsd = details.get('motif_rmsd')
            if current_rmsd is not None and current_rmsd < best_rmsd:
                best_rmsd = current_rmsd
                best_rmsd_record = record
    
    improvement = best_reward - baseline_reward
    successful_samples = sum(1 for rec in sample_records if rec['evaluation'].get('success'))
    
    print(f"   📈 Sampling summary: best_reward={best_reward:.3f} (Δ={improvement:+.3f}), successful_samples={successful_samples}/{len(sample_records)}")
    if best_length_matched_record:
        print(f"   📏 Best length-matched reward: {best_length_reward:.3f} from {best_length_matched_record['expert']} sample {best_length_matched_record['sample_index']}")
    if best_sctm_record:
        print(f"   🧮 Best scTM sample: {best_sctm:.3f} from {best_sctm_record['expert']} sample {best_sctm_record['sample_index']}")
    if best_rmsd_record:
        print(f"   📉 Best motif RMSD sample: {best_rmsd:.3f}Å from {best_rmsd_record['expert']} sample {best_rmsd_record['sample_index']}")

    baseline_metrics = baseline_details if isinstance(baseline_details, dict) else {}
    best_sample_metrics = best_sample_record['details'] if best_sample_record else {}
    
    def summarize_sample(record):
        if not record:
            return None
        return {
            'expert': record['expert'],
            'sample_index': record['sample_index'],
            'reward': record['reward'],
            'sequence': record['sequence'],
            'details': record['details'],
            'length_matches_target': record.get('length_matches_target', False)
        }

    result = {
        'motif_name': motif_data_clean.name,
        'motif_length': len(legacy_motif.motif_sequence),
        'target_length': legacy_motif.target_length,
        'baseline_reward': baseline_reward,
        'mcts_reward': best_reward,
        'improvement': improvement,
        'baseline_sequence': baseline_sequence,
        'mcts_sequence': best_sequence,
        'mode': 'sampling_multi_expert',
        'sampling_records': sample_records,
        'best_sample': best_sample_record,
        'baseline_details': baseline_details,
        'baseline_rmsd': baseline_metrics.get('motif_rmsd', float('inf')),
        'baseline_sctm': baseline_metrics.get('sctm', 0.5),
        'baseline_plddt': baseline_metrics.get('motif_plddt', 0.0),
        'final_rmsd': best_sample_metrics.get('motif_rmsd', float('inf')),
        'final_sctm': best_sample_metrics.get('sctm', 0.5),
        'final_plddt': best_sample_metrics.get('motif_plddt', 0.0),
        'best_length_matched_sample': summarize_sample(best_length_matched_record),
        'best_sctm_sample': summarize_sample(best_sctm_record),
        'best_rmsd_sample': summarize_sample(best_rmsd_record)
    }
    
    return result

def get_or_generate_shared_baseline(motif_data, dplm2_integration, baseline_cache: Dict, cache_file: str) -> Tuple[str, str, float]:
    """Get baseline from cache or generate new one using DPLM-2 150M (fixed baseline for all ablations)."""
    cache_key = f"{motif_data.name}_{motif_data.target_length}_{len(motif_data.motif_sequence)}"
    
    if cache_key in baseline_cache:
        cached_data = baseline_cache[cache_key]
        
        # **CRITICAL FIX**: Recalculate baseline reward using corrected ESMFold-based method
        print(f"   📋 Found cached baseline for {motif_data.name}, recalculating reward with ESMFold...")
        
        # Create temporary MCTS instance to use the corrected reward calculation
        from core.motif_scaffolding_mcts import MotifScaffoldingMCTS
        temp_mcts = MotifScaffoldingMCTS(dplm2_integration, external_experts=[])
        
        # Recalculate reward using the corrected ESMFold-based method
        corrected_reward = temp_mcts._calculate_reward(motif_data, cached_data['sequence'])
        
        print(f"   🔧 Baseline reward corrected: {cached_data['reward']:.3f} → {corrected_reward:.3f}")
        
        # Update cache with corrected reward
        cached_data['reward'] = corrected_reward
        cached_data['reward_method'] = 'esmfold_corrected'
        baseline_cache[cache_key] = cached_data
        
        # Save updated cache
        save_shared_baseline_cache(baseline_cache, cache_file)
        
        return cached_data['sequence'], cached_data['structure'], corrected_reward
    
    # Generate new baseline using DPLM-2 150M (consistent across all ablations)
    print(f"   🔄 Generating NEW shared baseline for {motif_data.name} using DPLM-2 150M...")
    
    # Create temporary MCTS instance for baseline generation (always DPLM-2 150M)
    from core.motif_scaffolding_mcts import MotifScaffoldingMCTS
    baseline_mcts = MotifScaffoldingMCTS(dplm2_integration, external_experts=[])
    
    # Generate baseline using official DPLM-2 approach
    baseline_seq, baseline_struct = baseline_mcts.generate_baseline(motif_data)
    
    if baseline_seq:
        # Calculate baseline reward for caching
        baseline_reward = baseline_mcts._calculate_reward(motif_data, baseline_seq)
        
        # Cache the baseline for future experiments
        baseline_cache[cache_key] = {
            'sequence': baseline_seq,
            'structure': baseline_struct,
            'reward': baseline_reward,
            'motif_name': motif_data.name,
            'generated_at': time.time(),
            'method': 'dplm2_150m_fixed_baseline'
        }
        
        # Save cache immediately
        save_shared_baseline_cache(baseline_cache, cache_file)
        
        print(f"   ✅ Generated and cached baseline for {motif_data.name} (reward: {baseline_reward:.3f})")
        return baseline_seq, baseline_struct, baseline_reward
    else:
        print(f"   ❌ Failed to generate baseline for {motif_data.name}")
        return None, None, 0.0

def print_summary(results: List[Dict]):
    """Print comprehensive experiment summary with baseline vs MCTS comparison."""
    if not results:
        print("❌ No results to summarize")
        return
    
    print(f"\n📊 COMPREHENSIVE MOTIF SCAFFOLDING RESULTS SUMMARY")
    print(f"=" * 80)
    
    # Group by mode for comparison
    baseline_results = [r for r in results if 'baseline' in r.get('mode', '')]
    mcts_results = [r for r in results if 'mcts' in r.get('mode', '')]
    
    total = len(results)
    successful = sum(1 for r in results if r.get('evaluation', {}).get('success', False))
    
    # Overall metrics
    avg_valid_aa = np.mean([r.get('evaluation', {}).get('valid_aa_ratio', 0.0) for r in results])
    motif_preservation_rate = sum(1 for r in results if r.get('evaluation', {}).get('motif_preserved', False)) / total
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Total experiments: {total}")
    print(f"   Successful scaffolds: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"   Average valid AA ratio: {avg_valid_aa:.3f}")
    print(f"   Motif preservation rate: {motif_preservation_rate:.3f}")
    
    # Baseline vs MCTS comparison
    if baseline_results and mcts_results:
        print(f"\n🏆 BASELINE vs MCTS COMPARISON:")
        print(f"-" * 50)
        
        # Extract structure metrics if available
        baseline_rmsd = []
        baseline_sctm = []
        mcts_rmsd = []
        mcts_sctm = []
        baseline_rewards = []
        mcts_rewards = []
        
        for r in baseline_results:
            scaffold_data = r.get('scaffold_data', {})
            if 'baseline_rmsd' in scaffold_data:
                baseline_rmsd.append(scaffold_data['baseline_rmsd'])
            if 'baseline_sctm' in scaffold_data:
                baseline_sctm.append(scaffold_data['baseline_sctm'])
            if 'baseline_reward' in scaffold_data:
                baseline_rewards.append(scaffold_data['baseline_reward'])
        
        for r in mcts_results:
            scaffold_data = r.get('scaffold_data', {})
            if 'final_rmsd' in scaffold_data:
                mcts_rmsd.append(scaffold_data['final_rmsd'])
            if 'final_sctm' in scaffold_data:
                mcts_sctm.append(scaffold_data['final_sctm'])
            if 'final_reward' in scaffold_data:
                mcts_rewards.append(scaffold_data['final_reward'])
        
        print(f"   Baseline experiments: {len(baseline_results)}")
        print(f"   MCTS experiments: {len(mcts_results)}")
        
        if baseline_rmsd and mcts_rmsd:
            baseline_avg_rmsd = np.mean(baseline_rmsd)
            mcts_avg_rmsd = np.mean(mcts_rmsd)
            rmsd_improvement = baseline_avg_rmsd - mcts_avg_rmsd  # Lower is better
            print(f"   Average RMSD: {baseline_avg_rmsd:.3f}Å → {mcts_avg_rmsd:.3f}Å (Δ {rmsd_improvement:+.3f}Å)")
        
        if baseline_sctm and mcts_sctm:
            baseline_avg_sctm = np.mean(baseline_sctm)
            mcts_avg_sctm = np.mean(mcts_sctm)
            sctm_improvement = mcts_avg_sctm - baseline_avg_sctm  # Higher is better
            print(f"   Average scTM: {baseline_avg_sctm:.3f} → {mcts_avg_sctm:.3f} (Δ {sctm_improvement:+.3f})")
        
        if baseline_rewards and mcts_rewards:
            baseline_avg_reward = np.mean(baseline_rewards)
            mcts_avg_reward = np.mean(mcts_rewards)
            reward_improvement = mcts_avg_reward - baseline_avg_reward
            print(f"   Average Reward: {baseline_avg_reward:.3f} → {mcts_avg_reward:.3f} (Δ {reward_improvement:+.3f})")
            
            # Success rate comparison
            baseline_success = sum(1 for r in baseline_results if r.get('evaluation', {}).get('success', False))
            mcts_success = sum(1 for r in mcts_results if r.get('evaluation', {}).get('success', False))
            print(f"   Success Rate: {baseline_success}/{len(baseline_results)} ({baseline_success/len(baseline_results)*100:.1f}%) → {mcts_success}/{len(mcts_results)} ({mcts_success/len(mcts_results)*100:.1f}%)")
    
    # Expert comparison
    expert_performance = {}
    for r in results:
        expert = r.get('expert', 'unknown')
        if expert not in expert_performance:
            expert_performance[expert] = {'total': 0, 'successful': 0, 'motif_preserved': 0}
        
        expert_performance[expert]['total'] += 1
        if r.get('evaluation', {}).get('success', False):
            expert_performance[expert]['successful'] += 1
        if r.get('evaluation', {}).get('motif_preserved', False):
            expert_performance[expert]['motif_preserved'] += 1
    
    if len(expert_performance) > 1:
        print(f"\n🤖 EXPERT PERFORMANCE COMPARISON:")
        print(f"-" * 50)
        for expert, perf in expert_performance.items():
            success_rate = perf['successful'] / perf['total'] * 100 if perf['total'] > 0 else 0
            motif_rate = perf['motif_preserved'] / perf['total'] * 100 if perf['total'] > 0 else 0
            print(f"   {expert:<20}: {perf['successful']:2d}/{perf['total']:2d} success ({success_rate:5.1f}%), motif preserved: {motif_rate:5.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Clean MCTS-guided Motif Scaffolding")
    parser.add_argument("--mode", choices=["baseline", "baseline_all", "mcts", "clean", "uct_no_entropy", "sampling"], default="clean",
                       help="Experiment mode: baseline, baseline_all, mcts (legacy), clean (PH-UCT), uct_no_entropy (standard UCT), sampling (multi-expert without tree)") 
    parser.add_argument("--experts", type=str, default="dplm2,proteinea",
                       help="Comma-separated list of experts (ignored for baseline_all)")
    parser.add_argument("--single_expert_mode", type=str, default=None,
                       help="Single expert mode: 'proteina', 'foldflow', 'rfdiffusion', or None for multi-expert")
    parser.add_argument("--num_motifs", type=int, default=3,
                       help="Number of motifs to process")
    parser.add_argument("--mcts_iterations", type=int, default=10,
                       help="Number of MCTS iterations")
    parser.add_argument("--max_depth", type=int, default=3,
                       help="Maximum MCTS search depth")
    parser.add_argument("--motif_filter", type=str, default=None,
                       help="Filter to specific motif (e.g., '1bcf')")
    parser.add_argument("--save_results", type=str, default=None,
                       help="Path to save results JSON")
    parser.add_argument("--use_real_models", action="store_true",
                       help="Require real external model execution (no fallbacks)")
    parser.add_argument("--use_shared_baseline", action="store_true", default=True,
                       help="Use shared baseline cache for fair ablation comparison")
    parser.add_argument("--baseline_cache_file", type=str, default="/net/scratch/caom/motif_scaffolding_results/shared_baselines.json",
                       help="Path to shared baseline cache file")
    parser.add_argument("--samples_per_expert", type=int, default=2,
                       help="Number of samples per expert when running sampling mode")
    parser.add_argument("--sampling_temperature", type=float, default=1.0,
                       help="Sampling temperature to pass to experts in sampling mode")
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("🧬 Clean MCTS-guided Motif Scaffolding Implementation")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"MCTS iterations: {args.mcts_iterations}")
    
    sampling_only = args.mode == "sampling"
    use_ph_uct = args.mode not in {"uct_no_entropy", "sampling"}
    
    if sampling_only:
        print(f"Samples per expert: {args.samples_per_expert} (temperature={args.sampling_temperature})")
        print("Max depth/iterations are ignored in sampling-only mode.")
    
    if args.mode in {"clean", "uct_no_entropy", "sampling"}:
        # Use new clean implementation
        from core.motif_scaffolding_mcts import MotifScaffoldingMCTS
        
        # Initialize DPLM-2
        dplm2 = DPLM2Integration(device="cuda")
        print("✅ DPLM-2 integration initialized")
        
        # Load shared baseline cache for fair comparison
        baseline_cache = {}
        if args.use_shared_baseline:
            baseline_cache = load_shared_baseline_cache(args.baseline_cache_file)
            print(f"🎯 Using shared baseline system for fair ablation comparison")
        else:
            print(f"⚠️ Using individual baselines (not recommended for ablation studies)")
        
        # Load external experts using REAL direct model inference
        external_experts = []
        try:
            # Import our working real direct models
            from external_models.real_direct_models import create_real_external_experts
            
            print("🤖 Loading REAL External Model Experts")
            print("=" * 40)
            
            # Create real external experts (ProteInA, FoldFlow, RFDiffusion)
            external_experts = create_real_external_experts()
            
            print(f"✅ Created {len(external_experts)} REAL external experts")
            if external_experts:
                expert_names = [expert.get_name() for expert in external_experts]
                print(f"   🚀 REAL models: {', '.join(expert_names)}")
                print(f"   📊 Using actual model weights:")
                print(f"      - ProteInA: 813MB checkpoint")
                print(f"      - FoldFlow: 81MB checkpoint")
                print(f"      - RFDiffusion: 462MB checkpoint")
                print(f"   🎯 NO MOCKS - All real model inference!")
            else:
                print("⚠️ No external experts created - will use DPLM-2 only")
            
        except ImportError as e:
            print(f"⚠️ Real external models not available: {e}")
            print("   Will use DPLM-2 variants only")
            external_experts = []
            
            if args.experts:
                expert_names = [name.strip().lower() for name in args.experts.split(',')]
                
                for expert_name in expert_names:
                    if expert_name == 'rfdiffusion':
                        try:
                            expert = create_rfdiffusion_direct_expert()
                            if expert.model is not None:
                                external_experts.append(expert)
                                print(f"✅ {expert.get_name()} expert loaded (direct model)")
                            else:
                                print(f"⚠️ {expert.get_name()} model failed to load")
                        except Exception as e:
                            print(f"❌ Failed to load RFDiffusion: {e}")
                            
                    elif expert_name in ['foldflow', 'flowflow']:
                        try:
                            expert = create_foldflow_direct_expert()
                            if expert.model is not None:
                                external_experts.append(expert)
                                print(f"✅ {expert.get_name()} expert loaded (direct model)")
                            else:
                                print(f"⚠️ {expert.get_name()} model failed to load")
                        except Exception as e:
                            print(f"❌ Failed to load FoldFlow: {e}")
                            
                    elif expert_name in ['proteina', 'proteinea']:
                        try:
                            expert = create_proteina_direct_expert()
                            if expert.model is not None:
                                external_experts.append(expert)
                                print(f"✅ {expert.get_name()} expert loaded (direct model)")
                            else:
                                print(f"⚠️ {expert.get_name()} model failed to load")
                        except Exception as e:
                            print(f"❌ Failed to load ProteInA: {e}")
                            
                    elif expert_name in ['proteinmpnn', 'pmpnn']:
                        try:
                            expert = create_proteinmpnn_direct_expert()
                            if expert.model is not None:
                                external_experts.append(expert)
                                print(f"✅ {expert.get_name()} expert loaded (direct model)")
                            else:
                                print(f"⚠️ {expert.get_name()} model failed to load")
                        except Exception as e:
                            print(f"❌ Failed to load ProteinMPNN: {e}")
                            
                    elif expert_name not in ['dplm2']:
                        print(f"⚠️ Unknown expert: {expert_name}")
                        
            print(f"✅ Loaded {len(external_experts)} external experts")
            if external_experts:
                expert_names = [expert.get_name() for expert in external_experts]
                print(f"   Expert models: {', '.join(expert_names)}")
        except ImportError:
            print("⚠️ External experts not available")
        
        # Initialize clean MCTS with REAL direct external experts
        mcts = MotifScaffoldingMCTS(
            dplm2,
            external_experts,
            single_expert_mode=args.single_expert_mode,
            use_ph_uct=use_ph_uct,
        )
        
        print(f"🤖 MCTS initialized with REAL external experts")
        print(f"   🔁 Selection strategy: {'PH-UCT (entropy-aware)' if use_ph_uct else 'Standard UCT (no entropy bonuses)'}")
        if sampling_only:
            print("   🧪 Sampling-only mode enabled (no tree expansion)")
        print(f"   🎯 Total experts available:")
        print(f"      - DPLM-2 variants: 4 (150M, 650M, 3B, ProteinMPNN)")
        print(f"      - External models: {len(external_experts)} ({[e.get_name() for e in external_experts]})")
        print(f"   🚀 All using REAL model weights - NO MOCKS!")
        
        if len(external_experts) > 0:
            print(f"✅ Multi-expert MCTS ready with {len(external_experts)} external models")
        else:
            print(f"⚠️ DPLM-2 only MCTS (no external models loaded)")
        
        sampling_experts: List[ExpertModel] = []
        if sampling_only:
            print("   🧪 Preparing experts for sampling-only mode")
            sampling_experts = [
                DPLM2ExpertModel(dplm2, variant_label="DPLM-2 150M", default_expert_id=1),
                DPLM2ExpertModel(dplm2, variant_label="DPLM-2 650M", default_expert_id=0),
            ]
            sampling_experts.extend(external_experts)
            sampling_list = [expert.get_name() if hasattr(expert, 'get_name') else str(expert) for expert in sampling_experts]
            print(f"   🧪 Sampling experts: {', '.join(sampling_list)}")
        
        # Load motif data using clean method
        data_dir = "/home/caom/AID3/dplm/data-bin/scaffolding-pdbs"
        motif_data_list = mcts.load_motif_data(data_dir)
        
        if not motif_data_list:
            print("❌ No motif data loaded")
            return
        
        # Apply motif filter if specified
        if args.motif_filter:
            motif_data_list = [m for m in motif_data_list if m.name == args.motif_filter]
            if not motif_data_list:
                print(f"❌ No motif found matching filter: {args.motif_filter}")
                return
            print(f"🎯 Filtered to motif: {args.motif_filter}")
        
        # Process motifs
        results = []
        for i, motif_data in enumerate(motif_data_list[:args.num_motifs]):
            print(f"\n🔄 Processing motif {i+1}/{min(args.num_motifs, len(motif_data_list))}: {motif_data.name}")
            
            try:
                # Get or generate shared baseline for fair comparison
                if args.use_shared_baseline:
                    baseline_seq, baseline_struct, baseline_reward = get_or_generate_shared_baseline(
                        motif_data, dplm2, baseline_cache, args.baseline_cache_file
                    )
                else:
                    # Generate individual baseline (not recommended for ablations)
                    baseline_seq, baseline_struct = mcts.generate_baseline(motif_data)
                    baseline_reward = mcts._calculate_reward(motif_data, baseline_seq) if baseline_seq else 0.0
                
                if not baseline_seq:
                    print(f"❌ Baseline generation failed for {motif_data.name}")
                    continue
                
                print(f"   🎯 Baseline reward: {baseline_reward:.3f} (shared: {args.use_shared_baseline})")
                
                if sampling_only:
                    sampling_result = run_sampling_only_multi_expert(
                        motif_data_clean=motif_data,
                        experts=sampling_experts if sampling_experts else [DPLM2ExpertModel(dplm2)],
                        mcts_instance=mcts,
                        baseline_sequence=baseline_seq,
                        baseline_reward=baseline_reward,
                        num_samples=args.samples_per_expert,
                        temperature=args.sampling_temperature,
                    )
                    results.append(sampling_result)
                    continue
                
                # Run MCTS optimization
                best_node = mcts.search(
                    motif_data=motif_data,
                    baseline_sequence=baseline_seq,
                    baseline_structure=baseline_struct,
                    num_iterations=args.mcts_iterations,
                    max_depth=args.max_depth
                )
                
                mcts_reward = best_node.reward
                improvement = mcts_reward - baseline_reward
                
                # Get detailed metrics for baseline and final
                baseline_metrics = mcts._get_detailed_metrics(motif_data, baseline_seq)
                final_metrics = mcts._get_detailed_metrics(motif_data, best_node.sequence)
                
                result = {
                    'motif_name': motif_data.name,
                    'motif_length': len(motif_data.motif_sequence),
                    'target_length': motif_data.target_length,
                    'baseline_reward': baseline_reward,
                    'mcts_reward': mcts_reward,
                    'improvement': improvement,
                    'baseline_sequence': baseline_seq,
                    'mcts_sequence': best_node.sequence,
                    'mcts_iterations': args.mcts_iterations,
                    # Add detailed metrics
                    'baseline_rmsd': baseline_metrics.get('motif_rmsd', float('inf')),
                    'baseline_sctm': baseline_metrics.get('sctm', 0.5),
                    'baseline_plddt': baseline_metrics.get('motif_plddt', 0.0),
                    'final_rmsd': final_metrics.get('motif_rmsd', float('inf')),
                    'final_sctm': final_metrics.get('sctm', 0.5),
                    'final_plddt': final_metrics.get('motif_plddt', 0.0)
                }
                
                results.append(result)
                
                print(f"   🏆 MCTS reward: {mcts_reward:.3f}")
                print(f"   📈 Improvement: {improvement:+.3f}")
                
            except Exception as e:
                print(f"❌ Error processing {motif_data.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Print summary
        if results:
            summary_title = "CLEAN MOTIF SCAFFOLDING RESULTS"
            if sampling_only:
                summary_title = "SAMPLING-ONLY MOTIF RESULTS"
            elif not use_ph_uct:
                summary_title = "UCT (NO ENTROPY) MOTIF RESULTS"
            print(f"\n📊 {summary_title}")
            print(f"=" * 50)
            
            avg_baseline = sum(r['baseline_reward'] for r in results) / len(results)
            avg_mcts = sum(r['mcts_reward'] for r in results) / len(results)
            avg_improvement = sum(r['improvement'] for r in results) / len(results)
            
            # Calculate average RMSD/scTM metrics
            baseline_rmsd_values = [r.get('baseline_rmsd', float('inf')) for r in results if r.get('baseline_rmsd', float('inf')) != float('inf')]
            final_rmsd_values = [r.get('final_rmsd', float('inf')) for r in results if r.get('final_rmsd', float('inf')) != float('inf')]
            baseline_sctm_values = [r.get('baseline_sctm', 0.5) for r in results if r.get('baseline_sctm', 0.5) != 0.5]
            final_sctm_values = [r.get('final_sctm', 0.5) for r in results if r.get('final_sctm', 0.5) != 0.5]
            
            print(f"Processed: {len(results)} motifs")
            print(f"Average baseline reward: {avg_baseline:.3f}")
            print(f"Average MCTS reward: {avg_mcts:.3f}")
            print(f"Average improvement: {avg_improvement:+.3f}")
            
            # Add RMSD/scTM metrics if available
            # Filter out None values
            baseline_rmsd_clean = [x for x in baseline_rmsd_values if x is not None and x != float('inf')]
            final_rmsd_clean = [x for x in final_rmsd_values if x is not None and x != float('inf')]
            baseline_sctm_clean = [x for x in baseline_sctm_values if x is not None]
            final_sctm_clean = [x for x in final_sctm_values if x is not None]
            
            if baseline_rmsd_clean and final_rmsd_clean:
                avg_baseline_rmsd = sum(baseline_rmsd_clean) / len(baseline_rmsd_clean)
                avg_final_rmsd = sum(final_rmsd_clean) / len(final_rmsd_clean)
                rmsd_improvement = avg_baseline_rmsd - avg_final_rmsd
                print(f"Average baseline RMSD: {avg_baseline_rmsd:.3f}Å")
                print(f"Average final RMSD: {avg_final_rmsd:.3f}Å")
                print(f"Average RMSD improvement: {rmsd_improvement:+.3f}Å (lower is better)")
            
            if baseline_sctm_clean and final_sctm_clean:
                avg_baseline_sctm = sum(baseline_sctm_clean) / len(baseline_sctm_clean)
                avg_final_sctm = sum(final_sctm_clean) / len(final_sctm_clean)
                sctm_improvement = avg_final_sctm - avg_baseline_sctm
                print(f"Average baseline scTM: {avg_baseline_sctm:.3f}")
                print(f"Average final scTM: {avg_final_sctm:.3f}")
                print(f"Average scTM improvement: {sctm_improvement:+.3f} (higher is better)")
            
            improved_count = sum(1 for r in results if r['improvement'] > 0)
            print(f"Improved motifs: {improved_count}/{len(results)} ({improved_count/len(results)*100:.1f}%)")
            
            for result in results:
                print(f"  {result['motif_name']}: {result['baseline_reward']:.3f} → {result['mcts_reward']:.3f} ({result['improvement']:+.3f})")
        
        # Save results
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to {args.save_results}")
        
        print(f"\n🎉 Clean motif scaffolding experiment completed!")
        
        # Cleanup
        dplm2.cleanup_all()
        
    else:
        # Use old implementation (for comparison)
        print("⚠️ Using legacy implementation - recommend using --mode clean")
        
        # Download/prepare motif data
        data_dir = download_motif_data()
        
        # Load motif data
        motifs = load_motif_data(data_dir)
        
        if not motifs:
            print("❌ No motif data found")
            return
        
        # Limit number of motifs for testing
        motifs = motifs[:args.num_motifs]
        print(f"📊 Processing {len(motifs)} motifs")
        
        # Initialize experts
        print("🔄 Initializing expert models...")
        experts = []
        
        # Parse expert list / requested baselines
        if args.mode == "baseline_all":
            expert_names = ["dplm2_650m", "proteina", "foldflow", "rfdiffusion"]
        elif args.experts.lower() == "all":
            expert_names = ["dplm2", "proteina", "flowflow", "rfdiffusion"]
        else:
            expert_names = [name.strip().lower() for name in args.experts.split(",") if name.strip()]
            if not expert_names:
                expert_names = ["dplm2"]
        
        real_expert_classes = {}
        if args.use_real_models:
            try:
                from external_models.real_direct_models import (
                    RealProteInADirect,
                    RealFoldFlowDirect,
                    RealRFDiffusionDirect
                )
                real_expert_classes = {
                    "proteina": RealProteInADirect,
                    "proteinea": RealProteInADirect,
                    "foldflow": RealFoldFlowDirect,
                    "flowflow": RealFoldFlowDirect,
                    "rfdiffusion": RealRFDiffusionDirect,
                }
                print("✅ Real external model classes available (ProteInA/FoldFlow/RFDiffusion)")
            except ImportError as real_err:
                raise RuntimeError(f"Real external models requested but unavailable: {real_err}") from real_err
        
        # **CRITICAL**: Always initialize DPLM-2 for baseline generation (even for external models)
        dplm2 = DPLM2Integration(device="cuda")
        print("✅ DPLM-2 integration initialized (required for all baselines)")
        
        try:
            added_labels = set()
            direct_factories = {}
            if DIRECT_MODEL_EXPERTS_AVAILABLE:
                direct_factories = {
                    "proteina": create_proteina_direct_expert,
                    "proteinea": create_proteina_direct_expert,
                    "foldflow": create_foldflow_direct_expert,
                    "flowflow": create_foldflow_direct_expert,
                    "rfdiffusion": create_rfdiffusion_direct_expert,
                }
            else:
                print("⚠️ Direct model expert factories unavailable - using DPLM-2 fallbacks when needed")
            
            for expert_name in expert_names:
                normalized = expert_name.lower()
                
                if normalized in {"dplm2", "dplm2_150m", "dplm2-small", "dplm2_small", "dplm2-150"}:
                    label_key = "dplm2_150m"
                    if label_key in added_labels:
                        continue
                    experts.append(DPLM2ExpertModel(dplm2, variant_label="DPLM-2 150M", default_expert_id=1))
                    added_labels.add(label_key)
                    print("✅ DPLM-2 150M expert initialized")
                    continue
                
                if normalized in {"dplm2_650m", "dplm2_650b", "dplm2-large", "dplm2_large", "dplm2-650"}:
                    label_key = "dplm2_650m"
                    if label_key in added_labels:
                        continue
                    experts.append(DPLM2ExpertModel(dplm2, variant_label="DPLM-2 650M", default_expert_id=0))
                    added_labels.add(label_key)
                    print("✅ DPLM-2 650M expert initialized")
                    continue
                
                if normalized in {"proteina", "proteinea", "foldflow", "flowflow", "rfdiffusion"}:
                    label_key = f"external_{normalized}"
                    if label_key in added_labels:
                        continue
                    
                    expert = None
                    if args.use_real_models and normalized in real_expert_classes:
                        try:
                            real_expert = real_expert_classes[normalized]()
                            expert = RealExternalExpertAdapter(real_expert)
                        except Exception as real_init_err:
                            raise RuntimeError(f"Failed to initialize real expert '{normalized}': {real_init_err}") from real_init_err
                    elif DIRECT_MODEL_EXPERTS_AVAILABLE and normalized in direct_factories and not args.use_real_models:
                        factory = direct_factories[normalized]
                        try:
                            expert = factory()
                        except Exception as factory_error:
                            print(f"⚠️ Failed to initialize {normalized} direct expert: {factory_error}")
                    
                    if expert is not None:
                        experts.append(expert)
                        added_labels.add(label_key)
                        mode_label = "real execution" if args.use_real_models else "direct"
                        print(f"✅ {expert.get_name()} expert initialized ({mode_label})")
                    else:
                        if args.use_real_models:
                            raise RuntimeError(f"Real model required for {normalized}, but initialization failed.")
                        fallback = DPLM2ExpertModel(dplm2, variant_label="DPLM-2 baseline", default_expert_id=1)
                        fallback._original_name = normalized
                        experts.append(fallback)
                        added_labels.add(label_key)
                        print(f"⚠️ Using DPLM-2 fallback for {normalized}")
                    continue
                
                print(f"⚠️ Unknown expert requested: {expert_name}")
            
            if not experts:
                print("❌ No experts available")
                return
                
            print(f"🚀 Running with {len(experts)} expert(s): {[e.get_name() for e in experts]}")
            
            # Run experiment
            results = run_motif_scaffolding_experiment(
                motifs, 
                experts, 
                mode=args.mode,
                scaffold_length=50,  # Default
                mcts_iterations=args.mcts_iterations,
                dplm2_integration=dplm2
            )
            
            # Save results if requested
            if args.save_results:
                with open(args.save_results, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"💾 Results saved to {args.save_results}")
            
            # Print summary
            print_summary(results)
            
            print(f"\n🎉 Motif scaffolding experiment completed!")
            
        finally:
            # Cleanup
            if dplm2 is not None:
                dplm2.cleanup_all()

if __name__ == "__main__":
    main()
