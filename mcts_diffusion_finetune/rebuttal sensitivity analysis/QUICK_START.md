# Sensitivity Analysis - Quick Start

## 🚀 Three Commands to Complete Analysis

### 1️⃣ Test (5 minutes)
```bash
cd /home/caom/AID3/dplm/mcts_diffusion_finetune/analysis
python test_sensitivity_analysis.py
```
✅ Validates pipeline with synthetic data

### 2️⃣ Analyze (10 minutes)
```bash
python reanalyze_existing_results.py \
  --results_dir /net/scratch/caom/mcts_results/cameo2022 \
  --output_dir ./sensitivity_analysis_cameo
```
✅ Reanalyzes existing MCTS results with 9 weight configurations

### 3️⃣ Review
```bash
# View Pareto fronts
eog ./sensitivity_analysis_cameo/pareto_fronts.png
eog ./sensitivity_analysis_cameo/pareto_front_3d.png

# Read report
cat ./sensitivity_analysis_cameo/sensitivity_analysis_report.txt
```
✅ Review visualizations and statistics

## 📊 What You Get

1. **pareto_fronts.png** - Six-panel visualization:
   - AAR vs scTM trade-offs
   - AAR vs Biophysical trade-offs  
   - scTM vs Biophysical trade-offs
   - Improvement space (Δ AAR vs Δ scTM)
   - Reward comparison by configuration
   - Per-metric heatmap

2. **pareto_front_3d.png** - 3D Pareto front (AAR vs scTM vs Bio)

3. **sensitivity_analysis_report.txt** - Comprehensive statistics:
   - Per-config mean ± std for AAR, scTM, Biophysical
   - Improvement rates
   - Key findings and trends

4. **sensitivity_analysis_detailed.json** - Raw data for further analysis

## 🎯 Weight Configurations Tested

| Name | AAR | scTM | Bio | Description |
|------|-----|------|-----|-------------|
| **baseline** | 0.60 | 0.35 | 0.05 | Original balanced |
| **sequence_dominant** | 0.80 | 0.15 | 0.05 | Strongly favor AAR |
| **sequence_focused** | 0.70 | 0.25 | 0.05 | Moderately favor AAR |
| **structure_dominant** | 0.15 | 0.80 | 0.05 | Strongly favor scTM |
| **structure_focused** | 0.25 | 0.70 | 0.05 | Moderately favor scTM |
| **equal_balance** | 0.475 | 0.475 | 0.05 | Equal AAR/scTM |
| **structure_slight_edge** | 0.40 | 0.55 | 0.05 | Slight scTM preference |
| **biophysical_aware** | 0.50 | 0.35 | 0.15 | Increased bio constraints |
| **biophysical_strict** | 0.45 | 0.30 | 0.25 | Strong bio constraints |

## 🔧 Batch Processing

For production runs:
```bash
sbatch analysis/run_cameo_sensitivity_analysis.sh \
  /net/scratch/caom/mcts_results/cameo2022 \
  ./sensitivity_analysis_cameo
```

## 📝 For the Paper

### Methods Section
```
To demonstrate robustness, we performed comprehensive sensitivity 
analyses over 9 reward weight configurations spanning sequence-first 
(AAR-dominant), structure-first (scTM-dominant), and balanced regimes. 
For each configuration, we report per-metric outcomes and visualize 
Pareto trade-offs, making all multi-objective trade-offs explicit.
```

### Supplementary Figure
Include `pareto_fronts.png` showing:
- 2D and 3D Pareto fronts
- Per-metric outcomes by configuration
- Improvement space visualization

## ❓ Troubleshooting

**No results loaded?**
- Check directory path
- Use `--convert_logs` for log files

**Missing scTM values?**
- Analyzer uses structural proxy if scTM unavailable

**Want different configs?**
- Edit `WEIGHT_CONFIGS` in `reward_weight_sensitivity_analysis.py`

## 📚 Full Documentation

See `README_SENSITIVITY_ANALYSIS.md` for comprehensive guide.
