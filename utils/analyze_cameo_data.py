#!/usr/bin/env python3
"""
CAMEO Data Structure Analysis Script

This script analyzes the raw CAMEO 2022 data files to understand:
1. What's in the pkl files exactly
2. Why amino acid index 20 appears
3. The structure of aatype arrays
4. How sequences are encoded
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from collections import Counter

def analyze_single_file(file_path: str) -> Dict[str, Any]:
    """Analyze a single CAMEO pkl file in detail."""
    print(f"\n{'='*60}")
    print(f"ANALYZING FILE: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        analysis = {
            'filename': os.path.basename(file_path),
            'keys': list(data.keys()),
            'data_types': {},
            'array_shapes': {},
            'aatype_analysis': {},
            'sequence_analysis': {}
        }
        
        # Analyze each key in the data
        print(f"📋 DATA KEYS: {list(data.keys())}")
        
        for key, value in data.items():
            print(f"\n🔍 KEY: {key}")
            print(f"   Type: {type(value)}")
            
            analysis['data_types'][key] = str(type(value))
            
            if isinstance(value, np.ndarray):
                print(f"   Shape: {value.shape}")
                print(f"   Dtype: {value.dtype}")
                analysis['array_shapes'][key] = value.shape
                
                if key == 'aatype':
                    # Deep analysis of aatype array
                    print(f"   🧬 AATYPE DETAILED ANALYSIS:")
                    print(f"      Min value: {np.min(value)}")
                    print(f"      Max value: {np.max(value)}")
                    print(f"      Unique values: {sorted(np.unique(value))}")
                    print(f"      Value counts: {dict(zip(*np.unique(value, return_counts=True)))}")
                    print(f"      First 20 values: {value[:20]}")
                    print(f"      Last 20 values: {value[-20:]}")
                    
                    # Check for patterns
                    value_counts = Counter(value)
                    print(f"      Most common values: {value_counts.most_common(5)}")
                    
                    # Check for index 20 specifically
                    index_20_count = np.sum(value == 20)
                    index_20_positions = np.where(value == 20)[0]
                    print(f"      Index 20 count: {index_20_count}")
                    print(f"      Index 20 positions: {index_20_positions[:10]}..." if len(index_20_positions) > 10 else f"      Index 20 positions: {index_20_positions}")
                    
                    # Check if index 20 is at start/end
                    if len(index_20_positions) > 0:
                        start_positions = index_20_positions[index_20_positions < 10]
                        end_positions = index_20_positions[index_20_positions >= len(value) - 10]
                        print(f"      Index 20 at start (first 10): {start_positions}")
                        print(f"      Index 20 at end (last 10): {end_positions}")
                    
                    analysis['aatype_analysis'] = {
                        'min_value': int(np.min(value)),
                        'max_value': int(np.max(value)),
                        'unique_values': sorted(np.unique(value).tolist()),
                        'value_counts': dict(zip(*np.unique(value, return_counts=True))),
                        'index_20_count': int(index_20_count),
                        'index_20_positions': index_20_positions.tolist()
                    }
                
                elif value.ndim == 1 and len(value) < 50:
                    print(f"   Values: {value}")
                elif value.ndim == 1:
                    print(f"   First 10: {value[:10]}")
                    print(f"   Last 10: {value[-10:]}")
                elif value.ndim == 2:
                    print(f"   First row: {value[0] if len(value) > 0 else 'Empty'}")
                    print(f"   Last row: {value[-1] if len(value) > 0 else 'Empty'}")
                elif value.ndim >= 3:
                    print(f"   First element shape: {value[0].shape if len(value) > 0 else 'Empty'}")
                    
            elif isinstance(value, (list, tuple)):
                print(f"   Length: {len(value)}")
                if len(value) > 0:
                    print(f"   First element type: {type(value[0])}")
                    if len(value) < 10:
                        print(f"   Values: {value}")
                    else:
                        print(f"   First 5: {value[:5]}")
                        print(f"   Last 5: {value[-5:]}")
                        
            elif isinstance(value, str):
                print(f"   Length: {len(value)}")
                print(f"   Content: {repr(value[:100])}{'...' if len(value) > 100 else ''}")
                
                if key == 'sequence':
                    # Analyze sequence string
                    print(f"   🧬 SEQUENCE DETAILED ANALYSIS:")
                    char_counts = Counter(value)
                    print(f"      Character counts: {dict(char_counts.most_common())}")
                    print(f"      Contains X: {'X' in value}")
                    print(f"      X count: {value.count('X')}")
                    print(f"      Starts with: {repr(value[:10])}")
                    print(f"      Ends with: {repr(value[-10:])}")
                    
                    analysis['sequence_analysis'] = {
                        'length': len(value),
                        'char_counts': dict(char_counts),
                        'contains_x': 'X' in value,
                        'x_count': value.count('X'),
                        'starts_with': value[:10],
                        'ends_with': value[-10:]
                    }
                    
            else:
                print(f"   Value: {value}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error analyzing {file_path}: {e}")
        return {'filename': os.path.basename(file_path), 'error': str(e)}

def compare_aatype_vs_sequence(data: Dict) -> None:
    """Compare aatype array with sequence string if both exist."""
    if 'aatype' in data and 'sequence' in data:
        aatype = data['aatype']
        sequence = data['sequence']
        
        print(f"\n🔄 COMPARING AATYPE vs SEQUENCE:")
        print(f"   AAType length: {len(aatype)}")
        print(f"   Sequence length: {len(sequence)}")
        print(f"   Length difference: {len(aatype) - len(sequence)}")
        
        # Map aatype to sequence
        aa_map = {
            0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G',
            8: 'H', 9: 'I', 10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S',
            16: 'T', 17: 'W', 18: 'Y', 19: 'V', 20: 'X'
        }
        
        # Convert aatype to sequence
        converted_seq = ''.join([aa_map.get(aa, '?') for aa in aatype])
        
        print(f"   Original sequence:  {sequence[:30]}...")
        print(f"   Converted from aatype: {converted_seq[:30]}...")
        
        # Check if they match
        if sequence == converted_seq:
            print(f"   ✅ Perfect match!")
        else:
            print(f"   ❌ Sequences don't match")
            # Find first difference
            for i, (orig, conv) in enumerate(zip(sequence, converted_seq)):
                if orig != conv:
                    print(f"   First difference at position {i}: '{orig}' vs '{conv}'")
                    break

def analyze_all_cameo_files(data_dir: str = "/net/scratch/caom/dplm_datasets/data-bin/cameo2022/preprocessed") -> None:
    """Analyze all CAMEO pkl files."""
    print(f"🔍 CAMEO DATA STRUCTURE ANALYSIS")
    print(f"📁 Data directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return
    
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    print(f"📊 Found {len(pkl_files)} pkl files")
    
    if not pkl_files:
        print(f"❌ No pkl files found in {data_dir}")
        return
    
    # Analyze first few files in detail
    detailed_files = pkl_files[:3]  # Analyze first 3 files in detail
    summary_files = pkl_files[3:8]  # Quick summary for next 5 files
    
    all_analyses = []
    
    print(f"\n🎯 DETAILED ANALYSIS OF FIRST 3 FILES:")
    for pkl_file in detailed_files:
        file_path = os.path.join(data_dir, pkl_file)
        analysis = analyze_single_file(file_path)
        all_analyses.append(analysis)
        
        # Load data for comparison
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            compare_aatype_vs_sequence(data)
        except Exception as e:
            print(f"❌ Error loading {pkl_file} for comparison: {e}")
    
    print(f"\n📋 QUICK SUMMARY OF NEXT 5 FILES:")
    for pkl_file in summary_files:
        file_path = os.path.join(data_dir, pkl_file)
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            aatype = data.get('aatype', [])
            sequence = data.get('sequence', '')
            
            print(f"   {pkl_file}:")
            print(f"      Keys: {list(data.keys())}")
            if len(aatype) > 0:
                unique_vals = np.unique(aatype)
                index_20_count = np.sum(aatype == 20) if isinstance(aatype, np.ndarray) else 0
                print(f"      AAType: length={len(aatype)}, unique={unique_vals}, index_20_count={index_20_count}")
            if sequence:
                x_count = sequence.count('X')
                print(f"      Sequence: length={len(sequence)}, X_count={x_count}")
                
        except Exception as e:
            print(f"   {pkl_file}: ❌ Error - {e}")
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    
    # Count files with index 20
    files_with_index_20 = 0
    files_with_x_in_sequence = 0
    
    for analysis in all_analyses:
        if 'aatype_analysis' in analysis and analysis['aatype_analysis'].get('index_20_count', 0) > 0:
            files_with_index_20 += 1
        if 'sequence_analysis' in analysis and analysis['sequence_analysis'].get('contains_x', False):
            files_with_x_in_sequence += 1
    
    print(f"   Files with index 20 in aatype: {files_with_index_20}/{len(all_analyses)}")
    print(f"   Files with X in sequence: {files_with_x_in_sequence}/{len(all_analyses)}")
    
    # Check if index 20 corresponds to special tokens
    print(f"\n🎯 HYPOTHESIS TESTING:")
    print(f"   Hypothesis 1: Index 20 represents start/end tokens")
    print(f"   Hypothesis 2: Index 20 represents padding tokens")
    print(f"   Hypothesis 3: Index 20 represents unknown/masked residues")
    
    for analysis in all_analyses:
        if 'aatype_analysis' in analysis:
            aa_analysis = analysis['aatype_analysis']
            positions = aa_analysis.get('index_20_positions', [])
            if positions:
                filename = analysis['filename']
                print(f"   {filename}: Index 20 at positions {positions[:5]}{'...' if len(positions) > 5 else ''}")
                
                # Check if at start/end
                total_length = len(positions) + aa_analysis.get('index_20_count', 0)  # This is wrong, let me fix
                # Actually, we need the total aatype length
                if 'array_shapes' in analysis and 'aatype' in analysis['array_shapes']:
                    total_length = analysis['array_shapes']['aatype'][0]
                    start_positions = [p for p in positions if p < 5]
                    end_positions = [p for p in positions if p >= total_length - 5]
                    
                    if start_positions:
                        print(f"      -> Likely START tokens at: {start_positions}")
                    if end_positions:
                        print(f"      -> Likely END tokens at: {end_positions}")
                    if not start_positions and not end_positions:
                        print(f"      -> Likely PADDING/UNKNOWN tokens (scattered)")

def create_visualization(data_dir: str = "/net/scratch/caom/dplm_datasets/data-bin/cameo2022/preprocessed") -> None:
    """Create visualizations of the data patterns."""
    try:
        import matplotlib.pyplot as plt
        
        pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')][:10]  # First 10 files
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CAMEO Data Analysis', fontsize=16)
        
        # Collect data
        sequence_lengths = []
        aatype_lengths = []
        index_20_counts = []
        x_counts = []
        
        for pkl_file in pkl_files:
            try:
                with open(os.path.join(data_dir, pkl_file), 'rb') as f:
                    data = pickle.load(f)
                
                if 'aatype' in data:
                    aatype = data['aatype']
                    aatype_lengths.append(len(aatype))
                    index_20_counts.append(np.sum(aatype == 20))
                
                if 'sequence' in data:
                    sequence = data['sequence']
                    sequence_lengths.append(len(sequence))
                    x_counts.append(sequence.count('X'))
                    
            except Exception as e:
                print(f"Error processing {pkl_file}: {e}")
        
        # Plot 1: Sequence vs AAType lengths
        axes[0, 0].scatter(sequence_lengths, aatype_lengths, alpha=0.7)
        axes[0, 0].plot([0, max(max(sequence_lengths), max(aatype_lengths))], 
                       [0, max(max(sequence_lengths), max(aatype_lengths))], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('Sequence Length')
        axes[0, 0].set_ylabel('AAType Length')
        axes[0, 0].set_title('Sequence vs AAType Lengths')
        
        # Plot 2: Index 20 counts
        axes[0, 1].hist(index_20_counts, bins=10, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Index 20 Count')
        axes[0, 1].set_ylabel('Number of Files')
        axes[0, 1].set_title('Distribution of Index 20 Counts')
        
        # Plot 3: X counts in sequences
        axes[1, 0].hist(x_counts, bins=10, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('X Count in Sequence')
        axes[1, 0].set_ylabel('Number of Files')
        axes[1, 0].set_title('Distribution of X Counts in Sequences')
        
        # Plot 4: Index 20 vs X counts
        axes[1, 1].scatter(index_20_counts, x_counts, alpha=0.7)
        axes[1, 1].set_xlabel('Index 20 Count (AAType)')
        axes[1, 1].set_ylabel('X Count (Sequence)')
        axes[1, 1].set_title('Index 20 vs X Counts')
        
        plt.tight_layout()
        output_path = '/home/caom/AID3/dplm/cameo_data_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved to: {output_path}")
        plt.close()
        
    except ImportError:
        print("⚠️ Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

if __name__ == "__main__":
    # Run the analysis
    analyze_all_cameo_files()
    
    # Create visualization
    create_visualization()
    
    print(f"\n✅ CAMEO data analysis complete!")
    print(f"🎯 Key findings will help us understand why index 20 appears in the data")
