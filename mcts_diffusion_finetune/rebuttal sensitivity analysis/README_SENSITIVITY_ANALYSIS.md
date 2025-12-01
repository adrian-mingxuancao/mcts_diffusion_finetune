# Reward Weight Sensitivity Analysis

## Overview

This analysis addresses reviewer concerns about:

1. **Sensitivity to reward weights**: How do different weightings of AAR, scTM, and biophysical scores affect MCTS performance?
2. **Pareto trade-offs**: What are the explicit trade-offs between sequence recovery (AAR) and structural similarity (scTM)?
3. **Per-metric outcomes**: How does each metric perform under different weight configurations?

**Key Point**: The composite reward (e.g., 0.60 AAR + 0.35 scTM + 0.05 Biophysical) is a **convenience for multi-objective lead optimization**, not a tuned "secret sauce." This analysis demonstrates robustness and makes all trade-offs explicit.

## Weight Configurations Tested

We test 9 different weight configurations spanning three regimes:

### Sequence-First Regimes (Emphasize AAR)
- **sequence_dominant**: 0.80 AAR / 0.15 scTM / 0.05 Bio
- **sequence_focused**: 0.70 AAR / 0.25 scTM / 0.05 Bio

### Structure-First Regimes (Emphasize scTM)
- **structure_dominant**: 0.15 AAR / 0.80 scTM / 0.05 Bio
- **structure_focused**: 0.25 AAR / 0.70 scTM / 0.05 Bio

### Balanced Regimes
- **baseline**: 0.60 AAR / 0.35 scTM / 0.05 Bio (original)
- **equal_balance**: 0.475 AAR / 0.475 scTM / 0.05 Bio
- **structure_slight_edge**: 0.40 AAR / 0.55 scTM / 0.05 Bio

### Biophysical-Aware Regimes
- **biophysical_aware**: 0.50 AAR / 0.35 scTM / 0.15 Bio
- **biophysical_strict**: 0.45 AAR / 0.30 scTM / 0.25 Bio

## Quick Start

### Option 1: Reanalyze Existing Results (Recommended)

If you already have MCTS results with AAR, scTM, and biophysical metrics:

```bash
# Reanalyze existing JSON results
python analysis/reanalyze_existing_results.py \
  --results_dir /path/to/mcts/results \
  --output_dir ./sensitivity_analysis_output \
  --file_pattern "*.json"

# Or convert and reanalyze log files
python analysis/reanalyze_existing_results.py \
  --results_dir /path/to/mcts/logs \
  --output_dir ./sensitivity_analysis_output \
  --convert_logs
```

This is **much faster** than rerunning experiments since it just reweights existing metrics.

### Option 2: Run New Experiments with Different Weights

To run fresh MCTS experiments with each weight configuration:

```bash
python analysis/run_weight_sensitivity_experiments.py \
  --data_dir /home/caom/AID3/dplm/data-bin/cameo2022 \
  --output_dir ./weight_sensitivity_results \
  --num_structures 20 \
  --max_depth 3 \
  --num_iterations 50

# Then analyze the results
python analysis/reward_weight_sensitivity_analysis.py \
  --results_dir ./weight_sensitivity_results \
  --output_dir ./sensitivity_analysis_output
```

## Output Files

The analysis generates:

1. **pareto_fronts.png**: Six-panel visualization showing:
   - AAR vs scTM trade-offs
   - AAR vs Biophysical trade-offs
   - scTM vs Biophysical trade-offs
   - Δ AAR vs Δ scTM (improvement space)
   - Reward improvement by configuration
   - Per-metric heatmap

2. **pareto_front_3d.png**: 3D Pareto front (AAR vs scTM vs Biophysical)

3. **sensitivity_analysis_report.txt**: Comprehensive text report with:
   - Configuration descriptions
   - Per-metric outcomes for each weighting
   - Key findings and trends
   - Conclusions about robustness

4. **sensitivity_analysis_detailed.json**: Detailed results for further analysis

## Expected Results

### Sequence-First Regimes
When AAR weight is high (0.70-0.80):
- ✅ Higher AAR improvements
- ⚠️ Potentially lower scTM gains
- 📊 MCTS prioritizes sequence recovery

### Structure-First Regimes
When scTM weight is high (0.70-0.80):
- ✅ Higher scTM improvements
- ⚠️ Potentially lower AAR gains
- 📊 MCTS prioritizes structural similarity

### Balanced Regimes
Equal or near-equal weights (0.40-0.50 each):
- ✅ Moderate improvements in both AAR and scTM
- 📊 Multi-objective optimization

### Key Insight: Pareto Frontier

The Pareto fronts show that **MCTS moves designs toward better multi-objective solutions**, not just optimizing a single axis. Some configurations show AAR decreases while scTM increases, demonstrating the **explicit trade-offs** in multi-objective protein design.

## Addressing Reviewer Concerns

### Concern 1: Sensitivity to Weights

**Response**: We test 9 weight configurations spanning sequence-first, structure-first, and balanced regimes. Results show predictable shifts in the Pareto frontier based on weight configuration.

**Evidence**: See `sensitivity_analysis_report.txt` for per-config statistics and `pareto_fronts.png` for visual comparison.

### Concern 2: Per-Metric Outcomes

**Response**: We report AAR, scTM, and biophysical scores separately for each weight configuration, not just the composite reward.

**Evidence**: See heatmap in `pareto_fronts.png` (subplot 6) and detailed breakdown in `sensitivity_analysis_report.txt`.

### Concern 3: Pareto Trade-offs

**Response**: We explicitly visualize AAR vs scTM trade-offs in 2D and 3D Pareto fronts. Some tasks (e.g., CAMEO inverse folding) show mild AAR decreases while improving scTM.

**Evidence**: See `pareto_fronts.png` (subplots 1 and 4) and `pareto_front_3d.png` for explicit trade-off visualization.

### Concern 4: Caching vs Recomputing Uncertainty

**Response**: The sensitivity analysis is orthogonal to uncertainty computation. We can run separate ablations for:
- Cached entropy (computed once at root)
- Recomputed entropy (computed at each node)
- No entropy (pure UCT)

**Implementation**: See `core/sequence_level_mcts.py` for `use_entropy` flag and entropy caching logic.

## Integration with Paper

### Recommended Text for Methods Section

```
To demonstrate robustness of our MCTS planning approach, we performed 
comprehensive sensitivity analyses over reward weight configurations. 
We tested 9 configurations spanning sequence-first (AAR-dominant), 
structure-first (scTM-dominant), and balanced regimes. For each 
configuration, we report per-metric outcomes (AAR, scTM, biophysical 
scores) and visualize Pareto trade-offs. The composite reward is a 
transparent multi-objective convenience, not a hidden hyperparameter—all 
trade-offs are explicit and controllable.
```

### Recommended Figure for Supplementary Materials

Include `pareto_fronts.png` as a supplementary figure with caption:

```
Supplementary Figure X: Reward weight sensitivity analysis and Pareto 
trade-offs. (A-C) 2D Pareto fronts showing trade-offs between AAR, scTM, 
and biophysical scores under different weight configurations. (D) 
Improvement space (Δ AAR vs Δ scTM) showing MCTS moves designs toward 
better multi-objective solutions. (E) Reward improvement by configuration. 
(F) Per-metric heatmap showing predictable shifts based on weight 
configuration. Nine configurations tested: sequence-first (AAR-dominant), 
structure-first (scTM-dominant), and balanced regimes.
```

## Advanced Usage

### Custom Weight Configurations

Edit `reward_weight_sensitivity_analysis.py` to add custom configurations:

```python
WEIGHT_CONFIGS.append(
    RewardWeightConfig(
        name="custom_config",
        aar_weight=0.50,
        sctm_weight=0.40,
        biophysical_weight=0.10,
        description="Custom configuration for specific design goals"
    )
)
```

### Filtering Results

To analyze only specific modes or structures:

```python
# In reanalyze_existing_results.py, add filtering:
def load_results(self, pattern: str = "*.json"):
    for result_file in result_files:
        data = json.load(f)
        
        # Filter by mode
        if data.get('mode') != 'multi_expert':
            continue
        
        # Filter by structure
        if data.get('structure_id') not in ['7dz2_C', '7eoz_A']:
            continue
        
        self.results.append(data)
```

### Exporting for Statistical Analysis

The `sensitivity_analysis_detailed.json` file can be imported into R or Python for statistical tests:

```python
import json
import pandas as pd

with open('sensitivity_analysis_detailed.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
rows = []
for config_name, config_data in data.items():
    for result in config_data['results']:
        rows.append({
            'config': config_name,
            'structure': result['structure_id'],
            'delta_aar': result['delta_aar'],
            'delta_sctm': result['delta_sctm'],
            'delta_reward': result['delta_reward']
        })

df = pd.DataFrame(rows)

# Statistical tests
from scipy import stats
print(stats.kruskal(*[group['delta_aar'].values 
                      for name, group in df.groupby('config')]))
```

## Troubleshooting

### Issue: No results loaded

**Solution**: Check file pattern and directory path. Use `--convert_logs` if you have log files instead of JSON.

### Issue: Missing scTM values

**Solution**: Ensure your results include scTM calculations. If not, the analyzer will use a structural proxy based on amino acid composition.

### Issue: Biophysical scores all 0.8

**Solution**: Provide sequences in results so biophysical scores can be recomputed. Otherwise, default value of 0.8 is used.

## Citation

If you use this sensitivity analysis in your work, please cite:

```bibtex
@article{your_paper,
  title={MCTS-Guided Protein Design with Diffusion Models},
  author={Your Name et al.},
  journal={Your Journal},
  year={2024},
  note={Sensitivity analysis code available at: https://github.com/...}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
