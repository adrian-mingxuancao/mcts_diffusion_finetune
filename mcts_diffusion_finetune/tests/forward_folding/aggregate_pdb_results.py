#!/usr/bin/env python3
"""
Aggregate PDB MCTS Folding Ablation Results from summary files
"""

import os
import re
import json
from collections import defaultdict
from pathlib import Path

def parse_summary_file(summary_path):
    """Parse a summary .txt file"""
    results = {}
    
    with open(summary_path, 'r') as f:
        content = f.read()
    
    # Find mode sections
    mode_sections = re.findall(
        r'Mode: (\w+)\n-+\n(.*?)(?=\nMode:|Avg RMSD|$)',
        content,
        re.DOTALL
    )
    
    for mode, section in mode_sections:
        # Parse structure lines
        lines = section.strip().split('\n')
        structures = []
        
        for line in lines[1:]:  # Skip header
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 10:
                try:
                    structures.append({
                        'structure': parts[0],
                        'length': int(parts[1]),
                        'base_rmsd': float(parts[2]),
                        'final_rmsd': float(parts[3]),
                        'rmsd_improvement': float(parts[4]),
                        'base_tmscore': float(parts[5]),
                        'final_tmscore': float(parts[6]),
                        'tmscore_improvement': float(parts[7]),
                        'improved': parts[8] == 'True',
                        'time': float(parts[9])
                    })
                except (ValueError, IndexError):
                    continue
        
        if structures:
            results[mode] = structures
    
    return results

def main():
    log_dir = Path("/home/caom/AID3/dplm/logs/mcts_folding_ablation")
    results_dir = Path("/net/scratch/caom/pdb_evaluation_results")
    
    # Find all PDB log files
    pdb_logs = sorted(log_dir.glob("mcts_fold_pdb_573302_*.out"))
    
    print(f"Found {len(pdb_logs)} PDB log files")
    
    # Group results by configuration
    config_results = defaultdict(list)
    
    for log_file in pdb_logs:
        # Extract task ID
        match = re.search(r'mcts_fold_pdb_\d+_(\d+)\.out', log_file.name)
        if not match:
            continue
        
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
        
        config_name = config_names[config_id]
        
        # Look for corresponding summary file
        with open(log_file, 'r', errors='ignore') as f:
            content = f.read()
        
        # Find summary file reference
        summary_match = re.search(r'Summary table saved → (.+\.txt)', content)
        if summary_match:
            summary_path = Path(summary_match.group(1))
            if summary_path.exists():
                results = parse_summary_file(summary_path)
                for mode, structures in results.items():
                    config_results[config_name].extend(structures)
                    print(f"  {log_file.name} → {config_name}: {len(structures)} structures")
    
    # Aggregate statistics
    print(f"\n{'='*120}")
    print("PDB MCTS Folding Ablation Analysis")
    print(f"{'='*120}\n")
    
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
            print(f"⚠️  No results for {config_name}")
            continue
        
        n = len(results)
        avg_base_rmsd = sum(r['base_rmsd'] for r in results) / n
        avg_final_rmsd = sum(r['final_rmsd'] for r in results) / n
        avg_rmsd_imp = sum(r['rmsd_improvement'] for r in results) / n
        
        avg_base_tm = sum(r['base_tmscore'] for r in results) / n
        avg_final_tm = sum(r['final_tmscore'] for r in results) / n
        avg_tm_imp = sum(r['tmscore_improvement'] for r in results) / n
        
        # Calculate composite reward: (1 - RMSD/20) * TM-score
        avg_base_reward = sum((1 - min(r['base_rmsd']/20, 1)) * r['base_tmscore'] for r in results) / n
        avg_final_reward = sum((1 - min(r['final_rmsd']/20, 1)) * r['final_tmscore'] for r in results) / n
        avg_reward_imp = avg_final_reward - avg_base_reward
        
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
            'avg_baseline_reward': avg_base_reward,
            'avg_final_reward': avg_final_reward,
            'avg_reward_improvement': avg_reward_imp,
            'improved_count': improved_count,
            'improved_percentage': improved_pct
        }
        
        print(f"Configuration: {config_name}")
        print(f"  Structures: {n}")
        print(f"  RMSD:       {avg_base_rmsd:.2f}Å → {avg_final_rmsd:.2f}Å (Δ{avg_rmsd_imp:+.2f}Å)")
        print(f"  TM-score:   {avg_base_tm:.3f} → {avg_final_tm:.3f} (Δ{avg_tm_imp:+.3f})")
        print(f"  Composite:  {avg_base_reward:.3f} → {avg_final_reward:.3f} (Δ{avg_reward_imp:+.3f})")
        print(f"  Improved: {improved_count}/{n} ({improved_pct:.1f}%)")
        print()
    
    # Save summary
    output_dir = Path("/net/scratch/caom/pdb_evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    json_file = output_dir / "pdb_folding_ablation_analysis.json"
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved JSON → {json_file}")
    
    # Save table
    table_file = output_dir / "pdb_folding_ablation_analysis.txt"
    with open(table_file, 'w') as f:
        f.write("PDB MCTS Folding Ablation Analysis\n")
        f.write("="*140 + "\n\n")
        
        f.write(f"{'Configuration':<25} {'N':<5} {'Base RMSD':<10} {'Final RMSD':<11} {'ΔRMSD':<9} "
                f"{'Base TM':<8} {'Final TM':<9} {'ΔTM':<9} "
                f"{'Base Comp':<10} {'Final Comp':<11} {'ΔComp':<9} {'Improved':<12}\n")
        f.write("-"*140 + "\n")
        
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
                   f"{s['avg_baseline_rmsd']:<10.2f} {s['avg_final_rmsd']:<11.2f} {s['avg_rmsd_improvement']:<+9.2f} "
                   f"{s['avg_baseline_tmscore']:<8.3f} {s['avg_final_tmscore']:<9.3f} {s['avg_tmscore_improvement']:<+9.3f} "
                   f"{s['avg_baseline_reward']:<10.3f} {s['avg_final_reward']:<11.3f} {s['avg_reward_improvement']:<+9.3f} "
                   f"{s['improved_count']}/{s['n_structures']} ({s['improved_percentage']:.1f}%)\n")
        
        f.write("\n")
        
        # Add comparison section
        f.write("\nKey Comparisons:\n")
        f.write("-"*80 + "\n\n")
        
        if "MCTS-PH (depth=5)" in summary and "Random (MCTS-0)" in summary:
            mcts_ph = summary["MCTS-PH (depth=5)"]
            random = summary["Random (MCTS-0)"]
            
            rmsd_gain = random['avg_final_rmsd'] - mcts_ph['avg_final_rmsd']
            tm_gain = mcts_ph['avg_final_tmscore'] - random['avg_final_tmscore']
            reward_gain = mcts_ph['avg_reward_improvement'] - random['avg_reward_improvement']
            
            f.write(f"MCTS-PH (depth=5) vs Random (MCTS-0):\n")
            f.write(f"  Final RMSD: {mcts_ph['avg_final_rmsd']:.2f}Å vs {random['avg_final_rmsd']:.2f}Å ({rmsd_gain:+.2f}Å better)\n")
            f.write(f"  Final TM-score: {mcts_ph['avg_final_tmscore']:.3f} vs {random['avg_final_tmscore']:.3f} ({tm_gain:+.3f} better)\n")
            f.write(f"  Composite improvement: {mcts_ph['avg_reward_improvement']:+.3f} vs {random['avg_reward_improvement']:+.3f} ({reward_gain:+.3f} better)\n\n")
        
        if "Single-Expert (650M)" in summary and "Single-Expert (150M)" in summary:
            expert_650 = summary["Single-Expert (650M)"]
            expert_150 = summary["Single-Expert (150M)"]
            
            rmsd_gain = expert_150['avg_final_rmsd'] - expert_650['avg_final_rmsd']
            tm_gain = expert_650['avg_final_tmscore'] - expert_150['avg_final_tmscore']
            reward_gain = expert_650['avg_reward_improvement'] - expert_150['avg_reward_improvement']
            
            f.write(f"Single-Expert 650M vs 150M:\n")
            f.write(f"  Final RMSD: {expert_650['avg_final_rmsd']:.2f}Å vs {expert_150['avg_final_rmsd']:.2f}Å ({rmsd_gain:+.2f}Å better)\n")
            f.write(f"  Final TM-score: {expert_650['avg_final_tmscore']:.3f} vs {expert_150['avg_final_tmscore']:.3f} ({tm_gain:+.3f} better)\n")
            f.write(f"  Composite improvement: {expert_650['avg_reward_improvement']:+.3f} vs {expert_150['avg_reward_improvement']:+.3f} ({reward_gain:+.3f} better)\n\n")
    
    print(f"📊 Saved table → {table_file}")
    
    # Print comparison
    print(f"\n{'='*80}")
    print("Key Comparisons")
    print(f"{'='*80}\n")
    
    if "MCTS-PH (depth=5)" in summary and "Random (MCTS-0)" in summary:
        mcts_ph = summary["MCTS-PH (depth=5)"]
        random = summary["Random (MCTS-0)"]
        
        rmsd_gain = random['avg_final_rmsd'] - mcts_ph['avg_final_rmsd']
        tm_gain = mcts_ph['avg_final_tmscore'] - random['avg_final_tmscore']
        reward_gain = mcts_ph['avg_reward_improvement'] - random['avg_reward_improvement']
        
        print(f"MCTS-PH vs Random:")
        print(f"  RMSD improvement: {rmsd_gain:+.2f}Å better")
        print(f"  TM-score improvement: {tm_gain:+.3f} better")
        print(f"  Composite reward gain: {reward_gain:+.3f} better")
        print()

if __name__ == "__main__":
    main()
