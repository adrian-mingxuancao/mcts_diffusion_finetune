#!/usr/bin/env python3
"""
Create Combined CAMEO Plots from Multiple JSON Files

This script combines multiple JSON files to create comprehensive plots showing
multi-expert vs single-expert performance across 183 CAMEO structures.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots with LARGE fonts
plt.style.use('default')
plt.rcParams.update({
    'font.size': 16,           # Increased from 12
    'axes.titlesize': 18,      # Increased from 14
    'axes.labelsize': 16,      # Increased from 12
    'xtick.labelsize': 14,     # Increased from 10
    'ytick.labelsize': 14,     # Increased from 10
    'legend.fontsize': 14,     # Increased from 11
    'figure.titlesize': 20,    # Increased from 16
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3
})

class CombinedCAMEOPlots:
    def __init__(self):
        self.output_dir = Path("three_metric_plot")
        self.output_dir.mkdir(exist_ok=True)
        
        # Define the files to combine
        self.multi_expert_files = [
            "/net/scratch/caom/cameo_evaluation_results/mcts_ablation_results_20251007_172823.json",
            "/net/scratch/caom/cameo_evaluation_results/mcts_ablation_results_20251007_134041.json", 
            "/net/scratch/caom/cameo_evaluation_results/mcts_ablation_results_20251007_134037.json",
            "/net/scratch/caom/cameo_evaluation_results/mcts_ablation_results_20251007_053332.json"
        ]
        
        self.single_expert_files = [
            "/net/scratch/caom/cameo_evaluation_results/mcts_ablation_results_20251007_200455.json",
            "/net/scratch/caom/cameo_evaluation_results/mcts_ablation_results_20251007_192039.json"
        ]
        
        self.data = []
        
    def load_and_combine_data(self):
        """Load and combine data from multiple JSON files."""
        print("📊 Loading and combining data from multiple files...")
        
        # Load multi-expert data
        for file_path in self.multi_expert_files:
            if os.path.exists(file_path):
                print(f"   Loading multi-expert data from: {os.path.basename(file_path)}")
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    multi_expert_count = sum(1 for item in file_data if item.get('mode') == 'multi_expert')
                    print(f"   Found {multi_expert_count} multi-expert records")
                    self.data.extend(file_data)
            else:
                print(f"   ⚠️ File not found: {file_path}")
        
        # Load single expert data
        for file_path in self.single_expert_files:
            if os.path.exists(file_path):
                print(f"   Loading single-expert data from: {os.path.basename(file_path)}")
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    single_expert_count = sum(1 for item in file_data if item.get('mode') in ['single_expert_0', 'single_expert_1'])
                    print(f"   Found {single_expert_count} single-expert records")
                    self.data.extend(file_data)
            else:
                print(f"   ⚠️ File not found: {file_path}")
        
        print(f"📊 Total records loaded: {len(self.data)}")
        
        # Count by mode
        mode_counts = {}
        for item in self.data:
            mode = item.get('mode', 'unknown')
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        print("📊 Records by mode:")
        for mode, count in mode_counts.items():
            print(f"   {mode}: {count}")
        
        return self.data
    
    def create_three_metric_plot(self, plot_type='bar'):
        """Create 1x3 plot showing AAR, Reward, and scTM improvements."""
        df = pd.DataFrame(self.data)
        df = df.fillna(0)
        
        # Filter for records with actual length data
        df_with_length = df[df['length'] > 0].copy()
        print(f"📊 Analyzing {len(df_with_length)} records with length data")
        
        # Define length categories
        df_with_length['length_category'] = pd.cut(df_with_length['length'],
                                                   bins=[0, 100, 200, 300, 400, float('inf')],
                                                   labels=['<100', '100-200', '200-300', '300-400', '>400'])
        
        # Separate single expert and multi-expert for comparison
        single_expert = df_with_length[df_with_length['mode'].isin(['single_expert_0', 'single_expert_1'])].copy()
        multi_expert = df_with_length[df_with_length['mode'] == 'multi_expert'].copy()
        
        print(f"📊 Single expert records: {len(single_expert)}")
        print(f"📊 Multi expert records: {len(multi_expert)}")
        
        if len(single_expert) == 0 or len(multi_expert) == 0:
            print("❌ Error: Need both single expert and multi expert data for comparison")
            return
        
        # Create 1x3 subplot with LARGER figure size
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # Increased from (18, 6)
        
        if plot_type == 'bar':
            fig.suptitle('Multi-Expert vs Single-Expert Performance (Combined CAMEO Data)', fontsize=20, fontweight='bold')
            self.create_bar_plots(axes, single_expert, multi_expert)
        else:  # box plot
            fig.suptitle('Multi-Expert vs Single-Expert Performance (Combined CAMEO Data - Box Plots)', fontsize=20, fontweight='bold')
            self.create_box_plots(axes, single_expert, multi_expert)
        
        # Adjust layout with more padding
        plt.tight_layout(pad=3.0)  # Increased padding
        
        # Save the plot with appropriate filename
        if plot_type == 'bar':
            plt.savefig(self.output_dir / 'combined_cameo_three_metric_comparison.pdf',
                       bbox_inches='tight', facecolor='white', dpi=300)
            plt.savefig(self.output_dir / 'combined_cameo_three_metric_comparison.png',
                       dpi=300, bbox_inches='tight', facecolor='white')
        else:  # box plot
            plt.savefig(self.output_dir / 'combined_cameo_three_metric_boxplot_comparison.pdf',
                       bbox_inches='tight', facecolor='white', dpi=300)
            plt.savefig(self.output_dir / 'combined_cameo_three_metric_boxplot_comparison.png',
                       dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
        
        # Print summary tables
        length_categories = ['<100', '100-200', '200-300', '300-400', '>400']
        self.print_summary_tables(single_expert, multi_expert, length_categories)
    
    def create_bar_plots(self, axes, single_expert, multi_expert):
        """Create bar plots for all three metrics."""
        length_categories = ['<100', '100-200', '200-300', '300-400', '>400']
        x_pos = np.arange(len(length_categories))
        width = 0.35
        
        # Define colors
        single_color = '#2E86AB'
        multi_color = '#A23B72'
        
        # Plot 1: AAR Improvement
        ax1 = axes[0]
        aar_single = single_expert.groupby('length_category')['aar_improvement'].mean()
        aar_multi = multi_expert.groupby('length_category')['aar_improvement'].mean()
        
        single_values = [aar_single.get(cat, 0) for cat in length_categories]
        multi_values = [aar_multi.get(cat, 0) for cat in length_categories]
        
        bars1 = ax1.bar(x_pos - width/2, single_values, width,
                       label='Single Expert', color=single_color, alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, multi_values, width,
                       label='Multi Expert', color=multi_color, alpha=0.8)
        
        ax1.set_xlabel('Protein Length (residues)', fontweight='bold')
        ax1.set_ylabel('AAR Improvement', fontweight='bold')
        ax1.set_title('(A) Amino Acid Recovery (AAR)', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(length_categories)
        ax1.legend()
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars with BETTER positioning
        y_max = max(max(single_values), max(multi_values))
        y_min = min(min(single_values), min(multi_values))
        y_range = y_max - y_min if y_max != y_min else 0.1
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                # Adjust label position to stay within bounds
                if height >= 0:
                    label_y = height + y_range * 0.02  # 2% above bar
                    va = 'bottom'
                else:
                    label_y = height - y_range * 0.02  # 2% below bar
                    va = 'top'
                
                ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                        f'{height:.3f}', ha='center', va=va, fontsize=12, fontweight='bold')
        
        # Set y-axis limits to accommodate labels
        ax1.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        # Plot 2: Reward Improvement
        ax2 = axes[1]
        reward_single = single_expert.groupby('length_category')['reward_improvement'].mean()
        reward_multi = multi_expert.groupby('length_category')['reward_improvement'].mean()
        
        single_values = [reward_single.get(cat, 0) for cat in length_categories]
        multi_values = [reward_multi.get(cat, 0) for cat in length_categories]
        
        bars1 = ax2.bar(x_pos - width/2, single_values, width,
                       label='Single Expert', color=single_color, alpha=0.8)
        bars2 = ax2.bar(x_pos + width/2, multi_values, width,
                       label='Multi Expert', color=multi_color, alpha=0.8)
        
        ax2.set_xlabel('Protein Length (residues)', fontweight='bold')
        ax2.set_ylabel('Reward Improvement', fontweight='bold')
        ax2.set_title('(B) Reward Improvement', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(length_categories)
        ax2.legend()
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars with BETTER positioning
        y_max = max(max(single_values), max(multi_values))
        y_min = min(min(single_values), min(multi_values))
        y_range = y_max - y_min if y_max != y_min else 0.1
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                # Adjust label position to stay within bounds
                if height >= 0:
                    label_y = height + y_range * 0.02  # 2% above bar
                    va = 'bottom'
                else:
                    label_y = height - y_range * 0.02  # 2% below bar
                    va = 'top'
                
                ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                        f'{height:.3f}', ha='center', va=va, fontsize=12, fontweight='bold')
        
        # Set y-axis limits to accommodate labels
        ax2.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        # Plot 3: scTM Improvement
        ax3 = axes[2]
        sctm_single = single_expert.groupby('length_category')['sctm_improvement'].mean()
        sctm_multi = multi_expert.groupby('length_category')['sctm_improvement'].mean()
        
        single_values = [sctm_single.get(cat, 0) for cat in length_categories]
        multi_values = [sctm_multi.get(cat, 0) for cat in length_categories]
        
        bars1 = ax3.bar(x_pos - width/2, single_values, width,
                       label='Single Expert', color=single_color, alpha=0.8)
        bars2 = ax3.bar(x_pos + width/2, multi_values, width,
                       label='Multi Expert', color=multi_color, alpha=0.8)
        
        ax3.set_xlabel('Protein Length (residues)', fontweight='bold')
        ax3.set_ylabel('scTM Improvement', fontweight='bold')
        ax3.set_title('(C) Structural Quality (scTM)', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(length_categories)
        ax3.legend()
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars with BETTER positioning
        y_max = max(max(single_values), max(multi_values))
        y_min = min(min(single_values), min(multi_values))
        y_range = y_max - y_min if y_max != y_min else 0.1
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                # Adjust label position to stay within bounds
                if height >= 0:
                    label_y = height + y_range * 0.02  # 2% above bar
                    va = 'bottom'
                else:
                    label_y = height - y_range * 0.02  # 2% below bar
                    va = 'top'
                
                ax3.text(bar.get_x() + bar.get_width()/2., label_y,
                        f'{height:.3f}', ha='center', va=va, fontsize=12, fontweight='bold')
        
        # Set y-axis limits to accommodate labels
        ax3.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
    
    def create_box_plots(self, axes, single_expert, multi_expert):
        """Create box plots for all three metrics."""
        # Define colors
        single_color = '#2E86AB'
        multi_color = '#A23B72'
        
        # Plot 1: AAR Improvement
        ax1 = axes[0]
        self.create_single_boxplot(ax1, single_expert, multi_expert, 'aar_improvement',
                                   '(A) Amino Acid Recovery (AAR)', single_color, multi_color)
        
        # Plot 2: Reward Improvement
        ax2 = axes[1]
        self.create_single_boxplot(ax2, single_expert, multi_expert, 'reward_improvement',
                                   '(B) Reward Improvement', single_color, multi_color)
        
        # Plot 3: scTM Improvement
        ax3 = axes[2]
        self.create_single_boxplot(ax3, single_expert, multi_expert, 'sctm_improvement',
                                   '(C) Structural Quality (scTM)', single_color, multi_color)
    
    def create_single_boxplot(self, ax, single_data, multi_data, metric, title, single_color, multi_color):
        """Create a single box plot for one metric."""
        # Prepare data for box plot
        plot_data = []
        labels = []
        
        length_categories = ['<100', '100-200', '200-300', '300-400', '>400']
        
        for cat in length_categories:
            # Single expert data
            single_cat_data = single_data[single_data['length_category'] == cat][metric].values
            if len(single_cat_data) > 0:
                plot_data.append(single_cat_data)
                labels.append(f'{cat}\n(Single)')
            
            # Multi expert data
            multi_cat_data = multi_data[multi_data['length_category'] == cat][metric].values
            if len(multi_cat_data) > 0:
                plot_data.append(multi_cat_data)
                labels.append(f'{cat}\n(Multi)')
        
        # Create box plot
        if plot_data:
            bp = ax.boxplot(plot_data, tick_labels=labels, patch_artist=True,
                           widths=0.6, showfliers=True, showmeans=True)
            
            # Color the boxes
            for i, patch in enumerate(bp['boxes']):
                if i % 2 == 0:  # Single expert boxes
                    patch.set_facecolor(single_color)
                    patch.set_alpha(0.7)
                else:  # Multi expert boxes
                    patch.set_facecolor(multi_color)
                    patch.set_alpha(0.7)
            
            # Color the medians
            for i, median in enumerate(bp['medians']):
                if i % 2 == 0:  # Single expert medians
                    median.set_color(single_color)
                    median.set_linewidth(2)
                else:  # Multi expert medians
                    median.set_color(multi_color)
                    median.set_linewidth(2)
        
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_ylabel('Improvement', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    def print_summary_tables(self, single_expert, multi_expert, length_categories):
        """Print summary tables for all three metrics."""
        print("\n📊 COMBINED CAMEO DATA SUMMARY TABLES:")
        print("=" * 80)
        
        # AAR Improvement
        aar_single = single_expert.groupby('length_category')['aar_improvement'].mean()
        aar_multi = multi_expert.groupby('length_category')['aar_improvement'].mean()
        
        print("\n🎯 AAR IMPROVEMENT:")
        print("-" * 50)
        print(f"{'Length Range':<12} {'Single Expert':<15} {'Multi Expert':<15} {'Advantage':<10}")
        print("-" * 60)
        for cat in length_categories:
            single_val = aar_single.get(cat, 0)
            multi_val = aar_multi.get(cat, 0)
            advantage = multi_val - single_val
            print(f"{cat:<12} {single_val:<15.3f} {multi_val:<15.3f} {advantage:<10.3f}")
        
        # Reward Improvement
        reward_single = single_expert.groupby('length_category')['reward_improvement'].mean()
        reward_multi = multi_expert.groupby('length_category')['reward_improvement'].mean()
        
        print("\n🎯 REWARD IMPROVEMENT:")
        print("-" * 50)
        print(f"{'Length Range':<12} {'Single Expert':<15} {'Multi Expert':<15} {'Advantage':<10}")
        print("-" * 60)
        for cat in length_categories:
            single_val = reward_single.get(cat, 0)
            multi_val = reward_multi.get(cat, 0)
            advantage = multi_val - single_val
            print(f"{cat:<12} {single_val:<15.3f} {multi_val:<15.3f} {advantage:<10.3f}")
        
        # scTM Improvement
        sctm_single = single_expert.groupby('length_category')['sctm_improvement'].mean()
        sctm_multi = multi_expert.groupby('length_category')['sctm_improvement'].mean()
        
        print("\n🧬 SCTM IMPROVEMENT:")
        print("-" * 50)
        print(f"{'Length Range':<12} {'Single Expert':<15} {'Multi Expert':<15} {'Advantage':<10}")
        print("-" * 60)
        for cat in length_categories:
            single_val = sctm_single.get(cat, 0)
            multi_val = sctm_multi.get(cat, 0)
            advantage = multi_val - single_val
            print(f"{cat:<12} {single_val:<15.3f} {multi_val:<15.3f} {advantage:<10.3f}")

def main():
    parser = argparse.ArgumentParser(description='Create combined CAMEO plots from multiple JSON files')
    parser.add_argument('--plot_type', choices=['bar', 'box'], default='bar',
                       help='Type of plot to create: bar or box')
    
    args = parser.parse_args()
    
    plotter = CombinedCAMEOPlots()
    
    print(f"🎨 Creating combined CAMEO {args.plot_type} plots...")
    print("=" * 70)
    
    # Load and combine data
    plotter.load_and_combine_data()
    
    # Create plots
    plotter.create_three_metric_plot(plot_type=args.plot_type)
    
    print(f"\n✅ Combined CAMEO {args.plot_type} plots completed!")
    print(f"📁 Output directory: {plotter.output_dir.absolute()}")

if __name__ == "__main__":
    main()
