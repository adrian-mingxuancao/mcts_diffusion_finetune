#!/usr/bin/env python3
"""
PDB Data Loader

Loads PDB_date dataset for MCTS ablation studies.
Similar to CAMEODataLoader but adapted for PDB_date structure.
"""

import os
import pickle
from typing import Dict, List, Optional, Any

class PDBDataLoader:
    """Data loader for PDB_date dataset."""
    
    def __init__(self, data_path: str = "/home/caom/AID3/dplm/data-bin/PDB_date"):
        self.data_path = data_path
        self.preprocessed_path = os.path.join(data_path, "preprocessed")
        self.aatype_fasta_path = os.path.join(data_path, "aatype.fasta")
        self.struct_fasta_path = os.path.join(data_path, "struct.fasta")
        
        # Load all structure files
        self.structures = self._load_structure_files()
        print(f"✅ Loaded {len(self.structures)} PDB structures")
        
        # Load reference sequences
        self.reference_sequences = self._load_reference_sequences()
        print(f"✅ Loaded {len(self.reference_sequences)} reference sequences")
        
        # Load structure sequences
        self.structure_sequences = self._load_structure_sequences()
        print(f"✅ Loaded {len(self.structure_sequences)} structure sequences")
    
    def _load_structure_files(self) -> List[str]:
        """Load all .pkl structure files from subdirectories."""
        structure_files = []
        
        # Get all subdirectories (a0, a1, a2, etc.)
        subdirs = [d for d in os.listdir(self.preprocessed_path) 
                  if os.path.isdir(os.path.join(self.preprocessed_path, d))]
        subdirs.sort()  # Sort for consistent ordering
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.preprocessed_path, subdir)
            pkl_files = [f for f in os.listdir(subdir_path) if f.endswith('.pkl')]
            pkl_files.sort()  # Sort for consistent ordering
            
            for pkl_file in pkl_files:
                # Store relative path from preprocessed directory
                relative_path = os.path.join(subdir, pkl_file)
                structure_files.append(relative_path)
        
        return structure_files
    
    def _load_reference_sequences(self) -> Dict[str, str]:
        """Load reference amino acid sequences from aatype.fasta."""
        sequences = {}
        
        if not os.path.exists(self.aatype_fasta_path):
            print(f"⚠️ Reference FASTA not found: {self.aatype_fasta_path}")
            return sequences
        
        with open(self.aatype_fasta_path, 'r') as f:
            current_id = None
            current_seq = []
            
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence
                    if current_id is not None:
                        sequences[current_id] = ''.join(current_seq)
                    
                    # Start new sequence
                    current_id = line[1:]  # Remove '>'
                    current_seq = []
                else:
                    current_seq.append(line)
            
            # Save last sequence
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
        
        return sequences
    
    def _load_structure_sequences(self) -> Dict[str, str]:
        """Load structure sequences from struct.fasta."""
        sequences = {}
        
        if not os.path.exists(self.struct_fasta_path):
            print(f"⚠️ Structure FASTA not found: {self.struct_fasta_path}")
            return sequences
        
        with open(self.struct_fasta_path, 'r') as f:
            current_id = None
            current_seq = []
            
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence
                    if current_id is not None:
                        sequences[current_id] = ','.join(current_seq)
                    
                    # Start new sequence
                    current_id = line[1:]  # Remove '>'
                    current_seq = []
                else:
                    # Split by comma and add to sequence
                    tokens = line.split(',')
                    current_seq.extend(tokens)
            
            # Save last sequence
            if current_id is not None:
                sequences[current_id] = ','.join(current_seq)
        
        return sequences
    
    def get_structure_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get structure data by index."""
        if index >= len(self.structures):
            return None
        
        structure_file = self.structures[index]
        structure_path = os.path.join(self.preprocessed_path, structure_file)
        
        if not os.path.exists(structure_path):
            print(f"⚠️ Structure file not found: {structure_path}")
            return None
        
        try:
            with open(structure_path, 'rb') as f:
                structure_data = pickle.load(f)
            return structure_data
        except Exception as e:
            print(f"❌ Error loading structure {structure_file}: {e}")
            return None
    
    def get_structure_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get structure data by name (e.g., '8A00')."""
        # Find the structure file with this name
        for structure_file in self.structures:
            if structure_file.endswith(f"{name}.pkl"):
                structure_path = os.path.join(self.preprocessed_path, structure_file)
                
                if not os.path.exists(structure_path):
                    continue
                
                try:
                    with open(structure_path, 'rb') as f:
                        structure_data = pickle.load(f)
                    return structure_data
                except Exception as e:
                    print(f"❌ Error loading structure {structure_file}: {e}")
                    continue
        
        return None
    
    def get_reference_sequence(self, name: str) -> Optional[str]:
        """Get reference amino acid sequence by name."""
        return self.reference_sequences.get(name)
    
    def get_structure_sequence(self, name: str) -> Optional[str]:
        """Get structure sequence by name."""
        return self.structure_sequences.get(name)
    
    def get_test_structure(self, index: int = 0) -> Dict[str, Any]:
        """Get a test structure for debugging."""
        structure_data = self.get_structure_by_index(index)
        if structure_data is None:
            # Return a dummy structure if loading fails
            return {
                "name": f"test_structure_{index}",
                "struct_seq": "159,162,163,164,165",
                "sequence": "IKKSI",
                "length": 5
            }
        
        # Get the structure name from the file path
        structure_file = self.structures[index]
        structure_name = os.path.splitext(os.path.basename(structure_file))[0]
        
        # Get structure sequence
        struct_seq = self.get_structure_sequence(structure_name)
        if struct_seq is None:
            struct_seq = "159,162,163,164,165"  # Fallback
        
        # Calculate length
        length = len(struct_seq.split(',')) if struct_seq else 5
        
        # Get reference sequence
        ref_seq = self.get_reference_sequence(structure_name)
        
        return {
            "name": f"PDB {structure_name}",
            "struct_seq": struct_seq,
            "sequence": ref_seq or "IKKSI",
            "length": length,
            "pdb_id": structure_name,
            "chain_id": "A",  # Default chain ID
            "structure_data": structure_data
        }
    
    def get_structure_info(self, index: int) -> Dict[str, Any]:
        """Get comprehensive structure information."""
        structure_file = self.structures[index]
        structure_name = os.path.splitext(os.path.basename(structure_file))[0]
        
        # Get structure data
        structure_data = self.get_structure_by_index(index)
        
        # Get sequences
        struct_seq = self.get_structure_sequence(structure_name)
        ref_seq = self.get_reference_sequence(structure_name)
        
        # Calculate length
        length = len(struct_seq.split(',')) if struct_seq else 0
        
        return {
            "index": index,
            "name": f"PDB {structure_name}",
            "pdb_id": structure_name,
            "chain_id": "A",
            "struct_seq": struct_seq,
            "reference_sequence": ref_seq,
            "length": length,
            "structure_file": structure_file,
            "structure_data": structure_data
        }
    
    def __len__(self) -> int:
        """Return number of structures."""
        return len(self.structures)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get structure by index."""
        return self.get_test_structure(index)

# Test the loader
if __name__ == "__main__":
    print("🧬 Testing PDB Data Loader")
    print("=" * 50)
    
    loader = PDBDataLoader()
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total structures: {len(loader)}")
    print(f"  Reference sequences: {len(loader.reference_sequences)}")
    print(f"  Structure sequences: {len(loader.structure_sequences)}")
    
    # Test loading a few structures
    print(f"\n🔍 Testing structure loading:")
    for i in range(min(3, len(loader))):
        structure = loader.get_test_structure(i)
        print(f"  Structure {i}: {structure['name']} (length: {structure['length']})")
    
    print(f"\n✅ PDB Data Loader test complete!")