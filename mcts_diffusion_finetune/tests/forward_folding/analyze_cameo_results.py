#!/usr/bin/env python3
"""
Analyze CAMEO MCTS Folding Ablation Results
Extract metrics from log files: RMSD, TM-score, Composite Reward
"""

import os
import re
import json
from collections import defaultdict
from pathlib import Path

def parse_log_file(log_path):
    """Extract results from a single log file"""
    results = []
    
    with open(log_path, 'r', errors='ignore') as f:
        content = f.read()
    
    # Find all structure processing blocks
    structure_blocks = re.findall(
        r'PROCESSING: (.*?)\n={70}.*?'
        r'Baseline RMSD: ([\d.]+)Å, TM-score: ([\d.]+).*?'
        r'Final RMSD: ([\d.]+)Å, TM-score: ([\d.]+).*?'
        r'Improvement: RMSD ([-+][\d.]+)Å, TM-score ([-+][\d.]+).*?'
        r'Composite reward: ([\d.]+)',
        content,
        re.DOTALL
    )
    
    for match in structure_blocks:
        structure_name, base_rmsd, base_tm, final_rmsd, final_tm, rmsd_imp, tm_imp, comp_reward = match
        
        results.append({
            'structure': structure_name.strip(),
            'baseline_rmsd': float(base_rmsd),
            'baseline_tmscore': float(base_tm),
            'final_rmsd': float(final_rmsd),
            'final_tmscore': float(final_tm),
            'rmsd_improvement': float(rmsd_imp),
            'tmscore_improvement': float(tm_imp),
            'composite_reward': float(comp_reward),
            'improved': float(rmsd_imp) < 0 or float(tm_imp) > 0
        })
    
    return results

def extract_config_from_filename(filename):
    """Extract configuration info from log filename"""
    # Format: mcts_fold_complete_JOBID_TASKID.out
    # Task ID maps to config: task_id % 7 = config_id
    match = re.search(r'mcts_fold_complete_\d+_(\d+)\.out', filename)
    if match:
        task_id = int(match.group(1))
        config_id = task_id % 7
        
        config_names = [
            "Random (MCTS-0)",
            "Single-Expert (150M)",
            "Single-Expert (650M)",
            "Single-Expert (3B)",
            "Sampling (depth=1)",
            "MCTS-PH (depth=5)",
            "MCTS-UCT (depth=5)"
        ]
        
        return config_id, config_names[config_id]
    return None, None

def main():
    log_dir = Path("/home/caom/AID3/dplm/logs/mcts_folding_ablation")
    
    # Find all CAMEO log files
    cameo_logs = sorted(log_dir.glob("mcts_fold_complete_570391_*.out"))
    
    print(f"Found {len(cameo_logs)} CAMEO log files")
    
    # Group results by configuration
    config_results = defaultdict(list)
    
    for log_file in cameo_logs:
        config_id, config_name = extract_config_from_filename(log_file.name)
        if config_id is None:
            continue
        
        results = parse_log_file(log_file)
        if results:
            config_results[config_name].extend(results)
            print(f"  {log_file.name}: {len(results)} structures → {config_name}")
    
    # Aggregate statistics
    print(f"\n{'='*80}")
    print("CAMEO MCTS Folding Ablation Analysis")
    print(f"{'='*80}\n")
    
    summary = {}
    
    for config_name in [
        "Random (MCTS-0)",
        "Single-Expert (150M)",
        "Single-Expert (650M)",
        "Single-Expert (3B)",
        "Sampling (depth=1)",
        "MCTS-PH (depth=5)",
        "MCTS-UCT (depth=5)"
    ]:
        results = config_results[config_name]
        if not results:
            continue
        
        n = len(results)
        avg_base_rmsd = sum(r['baseline_rmsd'] for r in results) / n
        avg_final_rmsd = sum(r['final_rmsd'] for r in results) / n
        avg_rmsd_imp = sum(r['rmsd_improvement'] for r in results) / n
        
        avg_base_tm = sum(r['baseline_tmscore'] for r in results) / n
        avg_final_tm = sum(r['final_tmscore'] for r in results) / n
        avg_tm_imp = sum(r['tmscore_improvement'] for r in results) / n
        
        avg_comp_reward = sum(r['composite_reward'] for r in results) / n
        
        improved_count = sum(1 for r in results if r['improved'])
        improved_pct = improved_count / n * 100
        
        summary[config_name] = {
            'n_structures': n,
            'avg_baseline_rmsd': avg_base_rmsd,
            'avg_final_rmsd': avg_final_rmsd,
            'avg_rmsd_improvement': avg_rmsd_imp,
            'avg_baseline_tmscore': avg_base_tm,
            'avg_final_tmscore': avg_final_tm,
            'avg_tmscore_improvement': avg_tm_imp,
            'avg_composite_reward': avg_comp_reward,
            'improved_count': improved_count,
            'improved_percentage': improved_pct
        }
        
        print(f"Configuration: {config_name}")
        print(f"  Structures: {n}")
        print(f"  RMSD:       {avg_base_rmsd:.2f}Å → {avg_final_rmsd:.2f}Å (Δ{avg_rmsd_imp:+.2f}Å)")
        print(f"  TM-score:   {avg_base_tm:.3f} → {avg_final_tm:.3f} (Δ{avg_tm_imp:+.3f})")
        print(f"  Composite Reward: {avg_comp_reward:.3f}")
        print(f"  Improved: {improved_count}/{n} ({improved_pct:.1f}%)")
        print()
    
    # Save summary
    output_dir = Path("/net/scratch/caom/cameo_evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    json_file = output_dir / "cameo_folding_ablation_analysis.json"
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved JSON summary → {json_file}")
    
    # Save detailed table
    table_file = output_dir / "cameo_folding_ablation_analysis.txt"
    with open(table_file, 'w') as f:
        f.write("CAMEO MCTS Folding Ablation Analysis\n")
        f.write("="*120 + "\n\n")
        
        f.write(f"{'Configuration':<25} {'N':<5} {'Base RMSD':<10} {'Final RMSD':<11} {'ΔRMSD':<8} "
                f"{'Base TM':<8} {'Final TM':<9} {'ΔTM':<8} {'Comp.Reward':<12} {'Improved':<10}\n")
        f.write("-"*120 + "\n")
        
        for config_name in [
            "Random (MCTS-0)",
            "Single-Expert (150M)",
            "Single-Expert (650M)",
            "Single-Expert (3B)",
            "Sampling (depth=1)",
            "MCTS-PH (depth=5)",
            "MCTS-UCT (depth=5)"
        ]:
            if config_name not in summary:
                continue
            
            s = summary[config_name]
            f.write(f"{config_name:<25} {s['n_structures']:<5} "
                   f"{s['avg_baseline_rmsd']:<10.2f} {s['avg_final_rmsd']:<11.2f} "
                   f"{s['avg_rmsd_improvement']:<+8.2f} "
                   f"{s['avg_baseline_tmscore']:<8.3f} {s['avg_final_tmscore']:<9.3f} "
                   f"{s['avg_tmscore_improvement']:<+8.3f} "
                   f"{s['avg_composite_reward']:<12.3f} "
                   f"{s['improved_count']}/{s['n_structures']} ({s['improved_percentage']:.1f}%)\n")
        
        f.write("\n")
    
    print(f"📊 Saved table summary → {table_file}")
    
    # Print comparison
    print(f"\n{'='*80}")
    print("Key Comparisons")
    print(f"{'='*80}\n")
    
    if "MCTS-PH (depth=5)" in summary and "Random (MCTS-0)" in summary:
        mcts_ph = summary["MCTS-PH (depth=5)"]
        random = summary["Random (MCTS-0)"]
        
        rmsd_gain = random['avg_final_rmsd'] - mcts_ph['avg_final_rmsd']
        tm_gain = mcts_ph['avg_final_tmscore'] - random['avg_final_tmscore']
        reward_gain = mcts_ph['avg_composite_reward'] - random['avg_composite_reward']
        
        print(f"MCTS-PH vs Random:")
        print(f"  RMSD improvement: {rmsd_gain:+.2f}Å better")
        print(f"  TM-score improvement: {tm_gain:+.3f} better")
        print(f"  Composite reward: {reward_gain:+.3f} better")
        print()

if __name__ == "__main__":
    main()
