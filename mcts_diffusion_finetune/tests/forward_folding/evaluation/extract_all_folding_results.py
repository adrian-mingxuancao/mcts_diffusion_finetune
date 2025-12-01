#!/usr/bin/env python3
"""
Extract and compare ALL folding results: deterministic, stochastic, and MCTS.
Calculates composite reward as: TM-score (for folding, no AAR component)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

# All model directories
MODELS = {
    # Deterministic (argmax decoding)
    "dplm2_150m": "/home/caom/AID3/dplm/generation-results/dplm2_150m/folding/forward_folding",
    "dplm2_650m": "/home/caom/AID3/dplm/generation-results/dplm2_650m/folding/forward_folding",
    "dplm2_3b": "/home/caom/AID3/dplm/generation-results/dplm2_3b/folding/forward_folding",
    # Stochastic (temperature-based sampling)
    # Note: Files are in folding/folding/forward_folding due to saveto path
    "dplm2_150m_stochastic": "/home/caom/AID3/dplm/generation-results/dplm2_150m_stochastic/folding/folding/forward_folding",
    "dplm2_650m_stochastic": "/home/caom/AID3/dplm/generation-results/dplm2_650m_stochastic/folding/folding/forward_folding",
    "dplm2_3b_stochastic": "/home/caom/AID3/dplm/generation-results/dplm2_3b_stochastic/folding/folding/forward_folding",
    # MCTS (multi-expert search)
    "mcts_me": "/home/caom/AID3/dplm/generation-results/mcts-me/folding/forward_folding",
}

def extract_per_protein_metrics(model_dir):
    """Extract per-protein metrics from all_top_samples.csv files (latest run only)."""
    results = []
    
    if not os.path.exists(model_dir):
        print(f"   ⚠️ Directory not found")
        return pd.DataFrame(results)
    
    # Find the latest run directory
    run_dirs = []
    for item in os.listdir(model_dir):
        if item.startswith("run_"):
            run_path = os.path.join(model_dir, item)
            if os.path.isdir(run_path):
                run_dirs.append(run_path)
    
    if not run_dirs:
        print(f"   ⚠️ No run directories found")
        return pd.DataFrame(results)
    
    # Get the latest run
    latest_run = sorted(run_dirs)[-1]
    print(f"   📁 Using latest run: {os.path.basename(latest_run)}")
    
    # Find all all_top_samples.csv files
    for root, dirs, files in os.walk(latest_run):
        if "all_top_samples.csv" in files:
            csv_path = os.path.join(root, "all_top_samples.csv")
            
            try:
                df = pd.read_csv(csv_path)
                protein_id = Path(root).parent.name
                
                if len(df) > 0:
                    row = df.iloc[0]
                    
                    result = {
                        'protein_id': protein_id,
                        'length': int(row['length']) if 'length' in row else None,
                        'bb_rmsd_to_gt': float(row['bb_rmsd_to_gt']) if 'bb_rmsd_to_gt' in row else None,
                        'bb_tmscore_to_gt': float(row['bb_tmscore_to_gt']) if 'bb_tmscore_to_gt' in row else None,
                        'mean_plddt': float(row['mean_plddt']) if 'mean_plddt' in row else None,
                        'helix_percent': float(row['helix_percent']) if 'helix_percent' in row else None,
                        'strand_percent': float(row['strand_percent']) if 'strand_percent' in row else None,
                    }
                    
                    # Composite reward for folding: R = α·TM + β·(1 - min(RMSD/10, 1)) + γ·pLDDT
                    # With pLDDT available: α=0.4, β=0.3, γ=0.3
                    # Without pLDDT: α=0.6, β=0.4
                    tm = result['bb_tmscore_to_gt']
                    rmsd = result['bb_rmsd_to_gt']
                    plddt = result['mean_plddt']
                    
                    if tm is not None:
                        tm_comp = min(max(tm, 0.0), 1.0)
                        rmsd_comp = (1.0 - min(rmsd / 10.0, 1.0)) if rmsd is not None and rmsd != float('inf') else 0.0
                        plddt_comp = min(max(plddt / 100.0, 0.0), 1.0) if plddt is not None else 0.0
                        
                        if plddt is not None:
                            # All components available
                            result['composite_reward'] = 0.4 * tm_comp + 0.3 * rmsd_comp + 0.3 * plddt_comp
                        elif rmsd is not None:
                            # TM and RMSD available
                            result['composite_reward'] = 0.6 * tm_comp + 0.4 * rmsd_comp
                        else:
                            # Only TM available
                            result['composite_reward'] = tm_comp
                    else:
                        result['composite_reward'] = None
                    
                    results.append(result)
                    
            except Exception as e:
                print(f"   ⚠️ Error processing {csv_path}: {e}")
                continue
    
    return pd.DataFrame(results)

def calculate_summary_statistics(df, model_name):
    """Calculate summary statistics for a model."""
    if len(df) == 0:
        return None
    
    summary = {
        'model': model_name,
        'num_proteins': len(df),
        'avg_length': df['length'].mean() if 'length' in df else None,
        # RMSD statistics
        'avg_rmsd': df['bb_rmsd_to_gt'].mean(),
        'median_rmsd': df['bb_rmsd_to_gt'].median(),
        'std_rmsd': df['bb_rmsd_to_gt'].std(),
        'min_rmsd': df['bb_rmsd_to_gt'].min(),
        'max_rmsd': df['bb_rmsd_to_gt'].max(),
        # TM-score statistics
        'avg_tmscore': df['bb_tmscore_to_gt'].mean(),
        'median_tmscore': df['bb_tmscore_to_gt'].median(),
        'std_tmscore': df['bb_tmscore_to_gt'].std(),
        'min_tmscore': df['bb_tmscore_to_gt'].min(),
        'max_tmscore': df['bb_tmscore_to_gt'].max(),
        # Composite reward (for folding: reward = TM-score)
        'avg_composite_reward': df['composite_reward'].mean(),
        'median_composite_reward': df['composite_reward'].median(),
        'std_composite_reward': df['composite_reward'].std(),
        'min_composite_reward': df['composite_reward'].min(),
        'max_composite_reward': df['composite_reward'].max(),
        # Other metrics
        'avg_plddt': df['mean_plddt'].mean() if 'mean_plddt' in df else None,
        # Success rates
        'tm_above_0.5': (df['bb_tmscore_to_gt'] > 0.5).sum(),
        'tm_above_0.7': (df['bb_tmscore_to_gt'] > 0.7).sum(),
        'tm_above_0.8': (df['bb_tmscore_to_gt'] > 0.8).sum(),
        'tm_above_0.9': (df['bb_tmscore_to_gt'] > 0.9).sum(),
        'rmsd_below_2': (df['bb_rmsd_to_gt'] < 2.0).sum(),
        'rmsd_below_5': (df['bb_rmsd_to_gt'] < 5.0).sum(),
        'rmsd_below_10': (df['bb_rmsd_to_gt'] < 10.0).sum(),
    }
    
    return summary

def main():
    print("="*80)
    print("COMPREHENSIVE FOLDING EVALUATION RESULTS")
    print("="*80)
    print()
    print("ℹ️  Composite Reward Formula:")
    print("   R_fold = 0.4·TM + 0.3·(1-min(RMSD/10,1)) + 0.3·pLDDT  (if pLDDT available)")
    print("   R_fold = 0.6·TM + 0.4·(1-min(RMSD/10,1))              (if no pLDDT)")
    print()
    
    all_results = {}
    all_summaries = []
    
    for model_name, model_dir in MODELS.items():
        print(f"📊 Processing {model_name}...")
        print(f"   Directory: {model_dir}")
        
        df = extract_per_protein_metrics(model_dir)
        
        if len(df) == 0:
            print(f"   ⚠️ No results found")
            print()
            continue
        
        print(f"   ✅ Extracted {len(df)} proteins")
        
        summary = calculate_summary_statistics(df, model_name)
        all_results[model_name] = df
        all_summaries.append(summary)
        
        print(f"   📈 RMSD: {summary['avg_rmsd']:.3f} ± {summary['std_rmsd']:.3f} Å (min={summary['min_rmsd']:.3f}, max={summary['max_rmsd']:.3f})")
        print(f"   📈 TM-score: {summary['avg_tmscore']:.4f} ± {summary['std_tmscore']:.4f} (min={summary['min_tmscore']:.4f}, max={summary['max_tmscore']:.4f})")
        print(f"   📈 Reward: {summary['avg_composite_reward']:.4f} ± {summary['std_composite_reward']:.4f} (min={summary['min_composite_reward']:.4f}, max={summary['max_composite_reward']:.4f})")
        print(f"   🎯 TM > 0.8: {summary['tm_above_0.8']}/{summary['num_proteins']} ({summary['tm_above_0.8']/summary['num_proteins']*100:.1f}%)")
        print()
    
    summary_df = pd.DataFrame(all_summaries)
    
    # Save results
    output_dir = "/home/caom/AID3/dplm/folding_evaluation_summary"
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, df in all_results.items():
        output_file = os.path.join(output_dir, f"{model_name}_per_protein.csv")
        df.to_csv(output_file, index=False)
        print(f"💾 Saved {model_name}: {output_file}")
    
    summary_file = os.path.join(output_dir, "all_models_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"💾 Saved summary: {summary_file}")
    
    summary_json = os.path.join(output_dir, "all_models_summary.json")
    summary_df.to_json(summary_json, orient='records', indent=2)
    print(f"💾 Saved JSON: {summary_json}")
    
    print()
    print("="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print()
    
    display_cols = ['model', 'num_proteins', 'avg_rmsd', 'std_rmsd', 'avg_tmscore', 
                    'std_tmscore', 'avg_composite_reward', 'tm_above_0.8']
    print(summary_df[display_cols].to_string(index=False))
    
    print()
    print("="*80)
    print("BEST MODELS")
    print("="*80)
    print()
    
    if len(summary_df) > 0:
        best_rmsd = summary_df.loc[summary_df['avg_rmsd'].idxmin()]
        best_tm = summary_df.loc[summary_df['avg_tmscore'].idxmax()]
        best_reward = summary_df.loc[summary_df['avg_composite_reward'].idxmax()]
        
        print(f"🏆 Best RMSD: {best_rmsd['model']} ({best_rmsd['avg_rmsd']:.3f} ± {best_rmsd['std_rmsd']:.3f} Å)")
        print(f"🏆 Best TM-score: {best_tm['model']} ({best_tm['avg_tmscore']:.4f} ± {best_tm['std_tmscore']:.4f})")
        print(f"🏆 Best Reward: {best_reward['model']} ({best_reward['avg_composite_reward']:.4f} ± {best_reward['std_composite_reward']:.4f})")
    
    print()
    print(f"📁 All results saved to: {output_dir}/")
    print("✅ Extraction complete!")

if __name__ == "__main__":
    main()
