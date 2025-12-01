#!/usr/bin/env python3
"""
Test script for sensitivity analysis with synthetic data.

This creates synthetic MCTS results to verify the analysis pipeline works.
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from reward_weight_sensitivity_analysis import SensitivityAnalyzer, WEIGHT_CONFIGS


def generate_synthetic_results(output_dir: Path, num_structures: int = 20):
    """Generate synthetic MCTS results for testing"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Generating {num_structures} synthetic results...")
    
    np.random.seed(42)
    
    for i in range(num_structures):
        structure_id = f"test_structure_{i:03d}"
        mode = 'multi_expert'  # Align synthetic data with MCTD-ME setting
        
        # Generate realistic baseline metrics
        baseline_aar = np.random.uniform(0.35, 0.65)
        baseline_sctm = np.random.uniform(0.40, 0.70)
        baseline_biophysical = np.random.uniform(0.75, 0.90)
        
        # Generate improvements (some positive, some negative)
        delta_aar = np.random.normal(0.01, 0.03)  # Mean +1% improvement
        delta_sctm = np.random.normal(0.02, 0.04)  # Mean +2% improvement
        delta_biophysical = np.random.normal(0.0, 0.02)  # Small changes
        
        # Compute final metrics
        final_aar = np.clip(baseline_aar + delta_aar, 0.0, 1.0)
        final_sctm = np.clip(baseline_sctm + delta_sctm, 0.0, 1.0)
        final_biophysical = np.clip(baseline_biophysical + delta_biophysical, 0.0, 1.0)
        
        # Generate synthetic sequences
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        seq_length = np.random.randint(100, 300)
        baseline_sequence = ''.join(np.random.choice(list(amino_acids), seq_length))
        
        # Mutate some positions for final sequence
        final_sequence = list(baseline_sequence)
        num_mutations = int(seq_length * (1.0 - final_aar))
        mutation_positions = np.random.choice(seq_length, num_mutations, replace=False)
        for pos in mutation_positions:
            final_sequence[pos] = np.random.choice(list(amino_acids))
        final_sequence = ''.join(final_sequence)
        
        # Create result dictionary
        result = {
            'structure_id': structure_id,
            'mode': mode,
            'baseline_aar': float(baseline_aar),
            'final_aar': float(final_aar),
            'baseline_sctm': float(baseline_sctm),
            'final_sctm': float(final_sctm),
            'baseline_biophysical': float(baseline_biophysical),
            'final_biophysical': float(final_biophysical),
            'baseline_sequence': baseline_sequence,
            'final_sequence': final_sequence,
            'baseline_reward': 0.6 * baseline_aar + 0.35 * baseline_sctm + 0.05 * baseline_biophysical,
            'final_reward': 0.6 * final_aar + 0.35 * final_sctm + 0.05 * final_biophysical
        }
        
        # Save result
        output_file = output_dir / f"{structure_id}_{mode}_results.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    print(f"✅ Generated {num_structures} synthetic results in: {output_dir}")


def test_sensitivity_analysis():
    """Test the sensitivity analysis pipeline"""
    print("\n" + "="*80)
    print("TESTING SENSITIVITY ANALYSIS PIPELINE")
    print("="*80 + "\n")
    
    # Create test directories
    test_data_dir = Path('./test_sensitivity_data')
    test_output_dir = Path('./test_sensitivity_output')
    
    # Generate synthetic results
    generate_synthetic_results(test_data_dir, num_structures=20)
    
    # Initialize analyzer
    print("\n📊 Initializing sensitivity analyzer...")
    analyzer = SensitivityAnalyzer(str(test_data_dir), str(test_output_dir))
    
    # Load results
    print("📂 Loading synthetic results...")
    analyzer.load_results(pattern='*.json')
    
    if not analyzer.results:
        print("❌ Failed to load results")
        return False
    
    print(f"✅ Loaded {len(analyzer.results)} results")
    
    # Test metric extraction
    print("\n🔍 Testing metric extraction...")
    test_result = analyzer.results[0]
    metrics = analyzer.extract_metrics(test_result)
    
    if metrics is None:
        print("❌ Failed to extract metrics")
        return False
    
    print(f"✅ Extracted metrics: AAR={metrics['final_aar']:.3f}, scTM={metrics['final_sctm']:.3f}")
    
    # Test reward recomputation
    print("\n🔄 Testing reward recomputation...")
    test_config = WEIGHT_CONFIGS[0]
    recomputed = analyzer.recompute_rewards_with_weights(metrics, test_config)
    
    print(f"✅ Recomputed reward with {test_config.name}: {recomputed['final_reward']:.3f}")
    
    # Run full sensitivity analysis
    print("\n📈 Running full sensitivity analysis...")
    sensitivity_results = analyzer.analyze_weight_sensitivity()
    
    if not sensitivity_results:
        print("❌ Sensitivity analysis failed")
        return False
    
    print(f"✅ Analyzed {len(sensitivity_results)} weight configurations")
    
    # Generate Pareto fronts
    print("\n📊 Generating Pareto fronts...")
    try:
        analyzer.generate_pareto_fronts(sensitivity_results)
        print("✅ Pareto fronts generated")
    except Exception as e:
        print(f"❌ Failed to generate Pareto fronts: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Generate summary report
    print("\n📝 Generating summary report...")
    try:
        analyzer.generate_summary_report(sensitivity_results)
        print("✅ Summary report generated")
    except Exception as e:
        print(f"❌ Failed to generate summary report: {e}")
        return False
    
    # Save detailed results
    print("\n💾 Saving detailed results...")
    try:
        analyzer.save_detailed_results(sensitivity_results)
        print("✅ Detailed results saved")
    except Exception as e:
        print(f"❌ Failed to save detailed results: {e}")
        return False
    
    # Verify output files
    print("\n🔍 Verifying output files...")
    expected_files = [
        'pareto_fronts.png',
        'pareto_front_3d.png',
        'sensitivity_analysis_report.txt',
        'sensitivity_analysis_detailed.json'
    ]
    
    all_exist = True
    for filename in expected_files:
        filepath = test_output_dir / filename
        if filepath.exists():
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} (missing)")
            all_exist = False
    
    if not all_exist:
        return False
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)
    print(f"\nTest outputs saved to: {test_output_dir}")
    print("\nYou can now run the analysis on real data:")
    print("  python analysis/reanalyze_existing_results.py \\")
    print("    --results_dir /path/to/real/results \\")
    print("    --output_dir ./sensitivity_analysis_output")
    print()
    
    return True


if __name__ == '__main__':
    success = test_sensitivity_analysis()
    sys.exit(0 if success else 1)
