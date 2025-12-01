#!/usr/bin/env python3
"""
Data loader for real protein evaluation datasets following DPLM-2's approach.

Based on DPLM-2's evaluation methodology using CAMEO 2022 and CATH datasets.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from Bio import SeqIO
import logging


logger = logging.getLogger(__name__)


class InverseFoldingDataLoader:
    """
    Data loader for inverse folding evaluation following DPLM-2's approach.
    
    Loads pre-tokenized structure data from FASTA files as used in DPLM-2 paper.
    """
    
    def __init__(self, data_dir: str = "data-bin"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing evaluation datasets
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
    def load_cameo2022_dataset(self) -> List[Dict[str, Any]]:
        """
        Load CAMEO 2022 dataset as used in DPLM-2 evaluation.
        
        Returns:
            List of protein structure data dictionaries
        """
        cameo_dir = os.path.join(self.data_dir, "cameo2022")
        struct_fasta = os.path.join(cameo_dir, "struct.fasta")
        aatype_fasta = os.path.join(cameo_dir, "aatype.fasta")
        
        if not os.path.exists(struct_fasta):
            self.logger.warning(f"CAMEO 2022 data not found at {struct_fasta}")
            return []
        
        return self._load_fasta_pair(struct_fasta, aatype_fasta)
    
    def load_pdb_date_dataset(self) -> List[Dict[str, Any]]:
        """
        Load PDB date split dataset as used in DPLM-2 evaluation.
        
        Returns:
            List of protein structure data dictionaries  
        """
        pdb_dir = os.path.join(self.data_dir, "PDB_date")
        struct_fasta = os.path.join(pdb_dir, "struct.fasta")
        aatype_fasta = os.path.join(pdb_dir, "aatype.fasta")
        
        if not os.path.exists(struct_fasta):
            self.logger.warning(f"PDB date data not found at {struct_fasta}")
            return []
        
        return self._load_fasta_pair(struct_fasta, aatype_fasta)
    
    def load_cath_dataset(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Load CATH 4.3 dataset as used in DPLM-2 evaluation.
        
        Args:
            split: Dataset split (train/valid/test)
            
        Returns:
            List of protein structure data dictionaries
        """
        cath_dir = os.path.join(self.data_dir, "cath_4.3")
        chain_set_jsonl = os.path.join(cath_dir, "chain_set.jsonl")
        chain_set_splits = os.path.join(cath_dir, "chain_set_splits.json")
        
        if not os.path.exists(chain_set_jsonl):
            self.logger.warning(f"CATH data not found at {chain_set_jsonl}")
            return []
        
        return self._load_cath_jsonl(chain_set_jsonl, chain_set_splits, split)
    
    def _load_fasta_pair(self, struct_fasta: str, aatype_fasta: str) -> List[Dict[str, Any]]:
        """
        Load paired structure and sequence FASTA files.
        
        Args:
            struct_fasta: Path to structure tokens FASTA
            aatype_fasta: Path to amino acid sequences FASTA
            
        Returns:
            List of data dictionaries
        """
        data = []
        
        # Load structure tokens
        struct_records = {}
        if os.path.exists(struct_fasta):
            for record in SeqIO.parse(struct_fasta, "fasta"):
                # Structure tokens are comma-separated in DPLM-2 format
                struct_tokens = str(record.seq).split(",")
                struct_records[record.id] = struct_tokens
        
        # Load amino acid sequences  
        aa_records = {}
        if os.path.exists(aatype_fasta):
            for record in SeqIO.parse(aatype_fasta, "fasta"):
                aa_records[record.id] = str(record.seq)
        
        # Combine structure and sequence data
        for protein_id in struct_records:
            if protein_id in aa_records:
                data.append({
                    'name': protein_id,
                    'sequence': aa_records[protein_id],
                    'structure_tokens': struct_records[protein_id],
                    'length': len(aa_records[protein_id])
                })
        
        self.logger.info(f"Loaded {len(data)} proteins from FASTA pair")
        return data
    
    def _load_cath_jsonl(self, jsonl_path: str, splits_path: str, split: str) -> List[Dict[str, Any]]:
        """
        Load CATH dataset from JSONL format.
        
        Args:
            jsonl_path: Path to chain_set.jsonl
            splits_path: Path to chain_set_splits.json  
            split: Dataset split to load
            
        Returns:
            List of data dictionaries
        """
        data = []
        
        # Load split information
        split_names = set()
        if os.path.exists(splits_path):
            with open(splits_path, 'r') as f:
                splits = json.load(f)
                split_names = set(splits.get(split, []))
        
        # Load protein data
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                name = entry['name']
                
                # Filter by split if specified
                if split_names and name not in split_names:
                    continue
                
                sequence = entry['seq']
                coords = entry['coords']
                
                # Convert coordinates to proper format
                structure_coords = {}
                for atom_type, atom_coords in coords.items():
                    structure_coords[atom_type] = np.asarray(atom_coords, dtype=np.float32)
                
                data.append({
                    'name': name,
                    'sequence': sequence,
                    'coords': structure_coords,
                    'length': len(sequence)
                })
        
        self.logger.info(f"Loaded {len(data)} proteins from CATH {split} split")
        return data
    
    def create_mock_structure_from_coords(self, coords: Dict[str, np.ndarray], 
                                        sequence: str) -> Dict[str, Any]:
        """
        Create mock structure compatible with MCTS framework from coordinates.
        
        Args:
            coords: Dictionary of atom coordinates by type
            sequence: Amino acid sequence
            
        Returns:
            Mock structure dictionary
        """
        length = len(sequence)
        
        # Extract CA coordinates for plDDT-like scoring
        ca_coords = coords.get('CA', np.random.randn(length, 3))
        
        # Mock plDDT scores based on coordinate quality
        plddt_scores = []
        for i in range(length):
            if i < len(ca_coords):
                # Use coordinate variance as proxy for confidence
                coord_variance = np.var(ca_coords[i])
                plddt = max(0.3, min(0.95, 0.8 - coord_variance * 0.1))
            else:
                plddt = 0.5  # Default confidence
            plddt_scores.append(plddt)
        
        return {
            'coordinates': ca_coords,
            'target_length': length,
            'plddt_scores': plddt_scores,
            'sequence': sequence,  # Reference sequence for validation
            'structure_type': 'real'
        }


def download_evaluation_data(data_dir: str = "data-bin"):
    """
    Download DPLM-2 evaluation datasets.
    
    Args:
        data_dir: Directory to save datasets
    """
    logger.info("To download DPLM-2 evaluation datasets:")
    logger.info("1. CATH datasets: bash scripts/download_cath.sh")
    logger.info("2. CAMEO 2022: Contact DPLM-2 authors for pre-tokenized data")
    logger.info("3. Alternative: Use small test set of your own proteins")
    
    # Create directory structure
    os.makedirs(os.path.join(data_dir, "cameo2022"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "PDB_date"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "cath_4.3"), exist_ok=True)
    
    logger.info(f"Created directory structure in {data_dir}")


def create_test_dataset() -> List[Dict[str, Any]]:
    """
    Create a small test dataset for development/testing.
    
    Returns:
        List of test protein data
    """
    test_proteins = [
        {
            'name': '1CRN',  # Crambin - 46 residues
            'sequence': 'TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN',
            'length': 46
        },
        {
            'name': '1UBQ',  # Ubiquitin - 76 residues  
            'sequence': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
            'length': 76
        },
        {
            'name': '1VII',  # Villin headpiece - 36 residues
            'sequence': 'MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
            'length': 36
        }
    ]
    
    # Add mock structure data
    for protein in test_proteins:
        length = protein['length']
        protein['structure_tokens'] = [f'{i%10}' for i in range(length)]  # Mock tokens
        protein['coords'] = {
            'CA': np.random.randn(length, 3),  # Mock CA coordinates
            'N': np.random.randn(length, 3),
            'C': np.random.randn(length, 3),
            'O': np.random.randn(length, 3)
        }
    
    return test_proteins


if __name__ == "__main__":
    # Example usage
    loader = InverseFoldingDataLoader()
    
    # Try to load real datasets
    cameo_data = loader.load_cameo2022_dataset()
    pdb_data = loader.load_pdb_date_dataset()
    cath_data = loader.load_cath_dataset("test")
    
    if not (cameo_data or pdb_data or cath_data):
        print("No real datasets found, creating test dataset...")
        test_data = create_test_dataset()
        print(f"Created {len(test_data)} test proteins")
    else:
        print(f"Loaded: CAMEO {len(cameo_data)}, PDB {len(pdb_data)}, CATH {len(cath_data)}")




