#!/usr/bin/env python3
"""
Aggregate CAMEO MCTS Folding Ablation Results from summary files
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

def extract_config_from_log(log_path):
    """Extract configuration from log file"""
    with open(log_path, 'r', errors='ignore') as f:
        first_lines = ''.join([next(f) for _ in range(50)])
    
    # Look for mode indicators
    if 'random_no_expert' in first_lines:
        return 'Random (MCTS-0)', 'random_no_expert'
    elif 'single_expert' in first_lines:
        if 'expert_id=0' in first_lines or '650M' in first_lines:
            return 'Single-Expert (650M)', 'single_expert_650M'
        elif 'expert_id=1' in first_lines or '150M' in first_lines:
            return 'Single-Expert (150M)', 'single_expert_150M'
        elif 'expert_id=2' in first_lines or '3B' in first_lines:
            return 'Single-Expert (3B)', 'single_expert_3B'
    elif 'multi_expert' in first_lines:
        if 'max_depth=1' in first_lines or 'depth 1' in first_lines:
            return 'Sampling (depth=1)', 'sampling'
        elif 'use_standard_uct' in first_lines or 'Standard UCT' in first_lines:
            return 'MCTS-UCT (depth=5)', 'mcts_uct'
        else:
            return 'MCTS-PH (depth=5)', 'mcts_ph'
    
    return None, None

def main():
    log_dir = Path("/home/caom/AID3/dplm/logs/mcts_folding_ablation")
    results_dir = Path("/net/scratch/caom/cameo_evaluation_results")
    
    # Find all summary files
    summary_files = sorted(results_dir.glob("mcts_folding_ablation_summary_*.txt"))
    
    print(f"Found {len(summary_files)} summary files")
    
    # Map summary files to their log files to determine config
    config_results = defaultdict(list)
    
    for summary_file in summary_files:
        # Extract timestamp from summary filename
        timestamp = summary_file.stem.replace('mcts_folding_ablation_summary_', '')
        
        # Find corresponding log file
        log_files = list(log_dir.glob(f"mcts_fold_complete_*_{timestamp[:8]}*.out"))
        
        if not log_files:
            # Try to find by checking all logs for this summary
            results = parse_summary_file(summary_file)
            for mode, structures in results.items():
                # Use mode name directly
                mode_map = {
                    'random_no_expert': 'Random (MCTS-0)',
                    'single_expert': 'Single-Expert',
                    'multi_expert': 'Multi-Expert',
                    'sampling': 'Sampling (depth=1)'
                }
                config_name = mode_map.get(mode, mode)
                config_results[config_name].extend(structures)
        else:
            # Determine config from log file
            config_name, mode_key = extract_config_from_log(log_files[0])
            if config_name:
                results = parse_summary_file(summary_file)
                for mode, structures in results.items():
                    config_results[config_name].extend(structures)
                print(f"  {summary_file.name} → {config_name}: {sum(len(s) for s in results.values())} structures")
    
    # Also parse from log files directly for task-based configs
    cameo_logs = sorted(log_dir.glob("mcts_fold_complete_570391_*.out"))
    
    for log_file in cameo_logs:
        # Extract task ID
        match = re.search(r'mcts_fold_complete_\d+_(\d+)\.out', log_file.name)
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
    print("CAMEO MCTS Folding Ablation Analysis")
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
    output_dir = Path("/net/scratch/caom/cameo_evaluation_results")
    
    # Save JSON
    json_file = output_dir / "cameo_folding_ablation_analysis.json"
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved JSON → {json_file}")
    
    # Save table
    table_file = output_dir / "cameo_folding_ablation_analysis.txt"
    with open(table_file, 'w') as f:
        f.write("CAMEO MCTS Folding Ablation Analysis\n")
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
    
    print(f"📊 Saved table → {table_file}")

if __name__ == "__main__":
    main()
