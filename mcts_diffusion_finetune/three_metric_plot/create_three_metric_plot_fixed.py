#!/usr/bin/env python3
"""
Create 1x3 Plot for AAR, Reward, and scTM Improvements - FIXED VERSION

This script creates a single PDF with three subplots showing multi-expert advantage
across protein lengths for AAR, Reward, and scTM metrics.

FIXES:
- Increased font sizes for better readability
- Fixed value label positioning to stay within figure bounds
- Improved layout and spacing
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

# Set style for publication-quality plots with MUCH LARGER fonts
plt.style.use('default')
# Force matplotlib to use larger fonts by setting them multiple times
plt.rcParams.update({
    'font.size': 24,           # Base font size
    'axes.titlesize': 28,      # Subplot titles
    'axes.labelsize': 24,      # Axis labels
    'xtick.labelsize': 20,     # X-axis tick labels
    'ytick.labelsize': 20,     # Y-axis tick labels
    'legend.fontsize': 22,     # Legend
    'figure.titlesize': 32,    # Main title
    'lines.linewidth': 3,
    'lines.markersize': 8,
    'axes.linewidth': 1.5,
    'grid.alpha': 0.3,
    'font.family': 'DejaVu Sans',
    'font.weight': 'bold'
})

# Force font settings again to ensure they stick
import matplotlib
matplotlib.rcParams['font.size'] = 24
matplotlib.rcParams['axes.titlesize'] = 28
matplotlib.rcParams['axes.labelsize'] = 24
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['legend.fontsize'] = 22

class ThreeMetricPlot:
    def __init__(self, results_file="/net/scratch/caom/extracted_combined_results_20250906_231516.json"):
        self.results_file = results_file
        self.output_dir = Path("three_metric_plot")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load results
        with open(self.results_file, 'r') as f:
            self.data = json.load(f)
        
        print(f"📊 Loaded {len(self.data)} results")
    
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
        
        # Separate single_expert_0 and multi-expert for direct comparison
        single_expert_0 = df_with_length[df_with_length['mode'] == 'single_expert_0'].copy()
        multi_expert = df_with_length[df_with_length['mode'] == 'multi_expert'].copy()
        
        print(f"📊 Single expert 0 records: {len(single_expert_0)}")
        print(f"📊 Multi expert records: {len(multi_expert)}")
        
        # Create 1x3 subplot with balanced figure size and spacing
        fig, axes = plt.subplots(1, 3, figsize=(30, 9.5))  # Slightly taller for better proportions
        
        if plot_type == 'bar':
            fig.suptitle('Multi-Expert Advantage Across Protein Lengths', fontsize=32, fontweight='bold', y=0.94)
            self.create_bar_plots(axes, single_expert_0, multi_expert)
        else:  # box plot
            fig.suptitle('Multi-Expert Advantage Across Protein Lengths (Box Plots)', fontsize=32, fontweight='bold', y=0.94)
            self.create_box_plots(axes, single_expert_0, multi_expert)
        
        # Apply balanced spacing with a bit more gap between title and plots
        plt.subplots_adjust(top=0.82, bottom=0.15, left=0.08, right=0.95, wspace=0.25)
        
        # Layout is already optimized with subplots_adjust above
        
        # Save the plot with appropriate filename
        if plot_type == 'bar':
            plt.savefig(self.output_dir / 'three_metric_comparison.pdf',
                       bbox_inches='tight', facecolor='white', dpi=300)
            plt.savefig(self.output_dir / 'three_metric_comparison.png',
                       dpi=300, bbox_inches='tight', facecolor='white')
        else:  # box plot
            plt.savefig(self.output_dir / 'three_metric_boxplot_comparison.pdf',
                       bbox_inches='tight', facecolor='white', dpi=300)
            plt.savefig(self.output_dir / 'three_metric_boxplot_comparison.png',
                       dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
        
        # Print summary tables
        length_categories = ['<100', '100-200', '200-300', '300-400', '>400']
        self.print_summary_tables(single_expert_0, multi_expert, length_categories)
    
    def print_summary_tables(self, single_expert_0, multi_expert, length_categories):
        """Print summary tables for all three metrics."""
        print("\n📊 SUMMARY TABLES:")
        print("=" * 80)
        
        # AAR Improvement
        aar_single = single_expert_0.groupby('length_category')['aar_improvement'].mean()
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
        reward_single = single_expert_0.groupby('length_category')['reward_improvement'].mean()
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
        sctm_single = single_expert_0.groupby('length_category')['sctm_improvement'].mean()
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
    
    def create_bar_plots(self, axes, single_expert_0, multi_expert):
        """Create bar plots for all three metrics."""
        length_categories = ['<100', '100-200', '200-300', '300-400', '>400']
        x_pos = np.arange(len(length_categories))
        width = 0.35
        
        # Define colors
        single_color = '#2E86AB'
        multi_color = '#A23B72'
        
        # Plot 1: AAR Improvement
        ax1 = axes[0]
        aar_single = single_expert_0.groupby('length_category')['aar_improvement'].mean()
        aar_multi = multi_expert.groupby('length_category')['aar_improvement'].mean()
        
        single_values = [aar_single.get(cat, 0) for cat in length_categories]
        multi_values = [aar_multi.get(cat, 0) for cat in length_categories]
        
        bars1 = ax1.bar(x_pos - width/2, single_values, width,
                       label='Single Expert', color=single_color, alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, multi_values, width,
                       label='Multi Expert', color=multi_color, alpha=0.8)
        
        ax1.set_xlabel('Protein Length (residues)', fontweight='bold', fontsize=24)
        ax1.set_ylabel('AAR Improvement', fontweight='bold', fontsize=24)
        ax1.set_title('(A) Amino Acid Recovery (AAR)', fontweight='bold', fontsize=28)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(length_categories, fontsize=20)
        ax1.legend(fontsize=22)
        ax1.tick_params(axis='y', labelsize=20)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars with BETTER positioning
        y_max = max(max(single_values), max(multi_values))
        y_min = min(min(single_values), min(multi_values))
        y_range = y_max - y_min
        
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
                        f'{height:.3f}', ha='center', va=va, fontsize=18, fontweight='bold')
        
        # Set y-axis limits to accommodate labels
        ax1.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        # Plot 2: Reward Improvement
        ax2 = axes[1]
        reward_single = single_expert_0.groupby('length_category')['reward_improvement'].mean()
        reward_multi = multi_expert.groupby('length_category')['reward_improvement'].mean()
        
        single_values = [reward_single.get(cat, 0) for cat in length_categories]
        multi_values = [reward_multi.get(cat, 0) for cat in length_categories]
        
        bars1 = ax2.bar(x_pos - width/2, single_values, width,
                       label='Single Expert', color=single_color, alpha=0.8)
        bars2 = ax2.bar(x_pos + width/2, multi_values, width,
                       label='Multi Expert', color=multi_color, alpha=0.8)
        
        ax2.set_xlabel('Protein Length (residues)', fontweight='bold', fontsize=24)
        ax2.set_ylabel('Reward Improvement', fontweight='bold', fontsize=24)
        ax2.set_title('(B) Reward Improvement', fontweight='bold', fontsize=28)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(length_categories, fontsize=20)
        ax2.legend(fontsize=22)
        ax2.tick_params(axis='y', labelsize=20)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars with BETTER positioning
        y_max = max(max(single_values), max(multi_values))
        y_min = min(min(single_values), min(multi_values))
        y_range = y_max - y_min
        
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
                        f'{height:.3f}', ha='center', va=va, fontsize=18, fontweight='bold')
        
        # Set y-axis limits to accommodate labels
        ax2.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        # Plot 3: scTM Improvement
        ax3 = axes[2]
        sctm_single = single_expert_0.groupby('length_category')['sctm_improvement'].mean()
        sctm_multi = multi_expert.groupby('length_category')['sctm_improvement'].mean()
        
        single_values = [sctm_single.get(cat, 0) for cat in length_categories]
        multi_values = [sctm_multi.get(cat, 0) for cat in length_categories]
        
        bars1 = ax3.bar(x_pos - width/2, single_values, width,
                       label='Single Expert', color=single_color, alpha=0.8)
        bars2 = ax3.bar(x_pos + width/2, multi_values, width,
                       label='Multi Expert', color=multi_color, alpha=0.8)
        
        ax3.set_xlabel('Protein Length (residues)', fontweight='bold', fontsize=24)
        ax3.set_ylabel('scTM Improvement', fontweight='bold', fontsize=24)
        ax3.set_title('(C) Structural Quality (scTM)', fontweight='bold', fontsize=28)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(length_categories, fontsize=20)
        ax3.legend(fontsize=22)
        ax3.tick_params(axis='y', labelsize=20)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars with BETTER positioning
        y_max = max(max(single_values), max(multi_values))
        y_min = min(min(single_values), min(multi_values))
        y_range = y_max - y_min
        
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
                        f'{height:.3f}', ha='center', va=va, fontsize=18, fontweight='bold')
        
        # Set y-axis limits to accommodate labels
        ax3.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
    
    def create_box_plots(self, axes, single_expert_0, multi_expert):
        """Create box plots for all three metrics."""
        # Define colors
        single_color = '#2E86AB'
        multi_color = '#A23B72'
        
        # Plot 1: AAR Improvement
        ax1 = axes[0]
        self.create_single_boxplot(ax1, single_expert_0, multi_expert, 'aar_improvement',
                                   '(A) Amino Acid Recovery (AAR)', single_color, multi_color)
        
        # Plot 2: Reward Improvement
        ax2 = axes[1]
        self.create_single_boxplot(ax2, single_expert_0, multi_expert, 'reward_improvement',
                                   '(B) Reward Improvement', single_color, multi_color)
        
        # Plot 3: scTM Improvement
        ax3 = axes[2]
        self.create_single_boxplot(ax3, single_expert_0, multi_expert, 'sctm_improvement',
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
        
        ax.set_title(title, fontsize=28, fontweight='bold')
        ax.set_ylabel('Improvement', fontsize=24)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45, labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

def main():
    parser = argparse.ArgumentParser(description='Create 1x3 plot for AAR, Reward, and scTM improvements')
    parser.add_argument('--results_file', default='/net/scratch/caom/extracted_combined_results_20250906_231516.json',
                       help='Path to results JSON file')
    parser.add_argument('--output_dir', default='three_metric_plot',
                       help='Output directory for plots')
    parser.add_argument('--plot_type', choices=['bar', 'box'], default='bar',
                       help='Type of plot to create: bar or box')
    
    args = parser.parse_args()
    
    plotter = ThreeMetricPlot(args.results_file)
    plotter.output_dir = Path(args.output_dir)
    plotter.output_dir.mkdir(exist_ok=True)
    
    print(f"🎨 Creating 1x3 {args.plot_type} plot for AAR, Reward, and scTM improvements...")
    print("=" * 70)
    
    plotter.create_three_metric_plot(plot_type=args.plot_type)
    
    print(f"\n✅ Three metric {args.plot_type} plot completed!")
    print(f"📁 Output directory: {plotter.output_dir.absolute()}")

if __name__ == "__main__":
    main()
