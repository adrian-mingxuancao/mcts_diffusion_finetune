#!/usr/bin/env python3
"""
Reanalyze Existing MCTS Results with Different Reward Weights

This script takes existing MCTS results (which contain AAR, scTM, and biophysical
metrics) and recomputes composite rewards under different weight configurations.

This is MUCH faster than rerunning MCTS experiments, since we already have the
per-metric values and just need to reweight them.
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from reward_weight_sensitivity_analysis import (
    WEIGHT_CONFIGS, 
    RewardWeightConfig,
    SensitivityAnalyzer
)


def extract_metrics_from_log(log_file: Path) -> Optional[Dict]:
    """
    Extract metrics from MCTS log files.
    
    Looks for patterns like:
    - "Baseline AAR: X%"
    - "Final AAR: X%"
    - "Baseline scTM: X"
    - "Final scTM: X"
    """
    metrics = {}
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract structure ID
        import re
        structure_match = re.search(r'Structure:\s+(\S+)', content)
        if structure_match:
            metrics['structure_id'] = structure_match.group(1)
        
        # Extract AAR values
        baseline_aar_match = re.search(r'Baseline AAR:\s+([\d.]+)%', content)
        final_aar_match = re.search(r'Final AAR:\s+([\d.]+)%', content)
        
        if baseline_aar_match and final_aar_match:
            metrics['baseline_aar'] = float(baseline_aar_match.group(1)) / 100.0
            metrics['final_aar'] = float(final_aar_match.group(1)) / 100.0
        
        # Extract scTM values
        baseline_sctm_match = re.search(r'Baseline scTM:\s+([\d.]+)', content)
        final_sctm_match = re.search(r'Final scTM:\s+([\d.]+)', content)
        
        if baseline_sctm_match and final_sctm_match:
            metrics['baseline_sctm'] = float(baseline_sctm_match.group(1))
            metrics['final_sctm'] = float(final_sctm_match.group(1))
        
        # Extract biophysical scores (if available)
        baseline_bio_match = re.search(r'Baseline Biophysical:\s+([\d.]+)', content)
        final_bio_match = re.search(r'Final Biophysical:\s+([\d.]+)', content)
        
        if baseline_bio_match and final_bio_match:
            metrics['baseline_biophysical'] = float(baseline_bio_match.group(1))
            metrics['final_biophysical'] = float(final_bio_match.group(1))
        else:
            # Use default biophysical scores
            metrics['baseline_biophysical'] = 0.8
            metrics['final_biophysical'] = 0.8
        
        # Extract sequences (for biophysical recomputation if needed)
        baseline_seq_match = re.search(r'Baseline sequence:\s+(\S+)', content)
        final_seq_match = re.search(r'Final sequence:\s+(\S+)', content)
        
        if baseline_seq_match:
            metrics['baseline_sequence'] = baseline_seq_match.group(1)
        if final_seq_match:
            metrics['final_sequence'] = final_seq_match.group(1)
        
        # Check if we have minimum required metrics
        if 'baseline_aar' in metrics and 'final_aar' in metrics:
            return metrics
        else:
            return None
            
    except Exception as e:
        print(f"⚠️ Failed to parse {log_file}: {e}")
        return None


def convert_existing_results_to_standard_format(
    results_dir: Path,
    output_dir: Path,
    file_pattern: str = "*.log"
) -> int:
    """
    Convert existing log files to standard JSON format.
    
    Args:
        results_dir: Directory containing log files
        output_dir: Directory to save converted JSON files
        file_pattern: Pattern to match log files
    
    Returns:
        Number of files converted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_files = list(results_dir.glob(file_pattern))
    print(f"📂 Found {len(log_files)} log files")
    
    converted = 0
    
    for log_file in log_files:
        metrics = extract_metrics_from_log(log_file)
        
        if metrics is None:
            continue
        
        # Save as JSON
        output_file = output_dir / f"{log_file.stem}_converted.json"
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        converted += 1
    
    print(f"✅ Converted {converted} log files to JSON")
    return converted


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reanalyze existing MCTS results with different reward weights"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing MCTS result files (JSON or logs)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./sensitivity_analysis_output',
        help='Directory to save analysis outputs'
    )
    parser.add_argument(
        '--file_pattern',
        type=str,
        default='*.json',
        help='File pattern to match (*.json or *.log)'
    )
    parser.add_argument(
        '--convert_logs',
        action='store_true',
        help='Convert log files to JSON format first'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    print("\n" + "="*80)
    print("REANALYZING EXISTING RESULTS WITH DIFFERENT REWARD WEIGHTS")
    print("="*80 + "\n")
    
    # Convert logs if requested
    if args.convert_logs:
        print("🔄 Converting log files to JSON format...")
        converted_dir = results_dir / 'converted_json'
        num_converted = convert_existing_results_to_standard_format(
            results_dir, converted_dir, file_pattern="*.log"
        )
        
        if num_converted > 0:
            print(f"✅ Using converted files from: {converted_dir}")
            results_dir = converted_dir
            args.file_pattern = '*.json'
        else:
            print("⚠️ No files converted, using original directory")
    
    # Initialize analyzer
    analyzer = SensitivityAnalyzer(str(results_dir), str(output_dir))
    
    # Load results
    analyzer.load_results(pattern=args.file_pattern)
    
    if not analyzer.results:
        print("❌ No results loaded. Exiting.")
        print("\nTip: If you have log files, use --convert_logs flag")
        return
    
    # Analyze sensitivity to weights
    sensitivity_results = analyzer.analyze_weight_sensitivity()
    
    if not sensitivity_results:
        print("❌ No sensitivity results generated. Check your data.")
        return
    
    # Generate Pareto fronts
    analyzer.generate_pareto_fronts(sensitivity_results)
    
    # Generate summary report
    analyzer.generate_summary_report(sensitivity_results)
    
    # Save detailed results
    analyzer.save_detailed_results(sensitivity_results)
    
    print("\n" + "="*80)
    print("✅ REANALYSIS COMPLETE")
    print(f"📁 All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - pareto_fronts.png: 2D Pareto front visualizations")
    print("  - pareto_front_3d.png: 3D Pareto front (AAR vs scTM vs Bio)")
    print("  - sensitivity_analysis_report.txt: Comprehensive text report")
    print("  - sensitivity_analysis_detailed.json: Detailed results for further analysis")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
