# Weight Sensitivity Analysis: Documentation Guide

## 📁 Files Overview

This directory contains comprehensive analysis of the MCTS reward weight sensitivity experiment with **two analysis versions**:

1. **All Available Structures**: N=15-25 per configuration (total 180 results)
2. **Shared Structures Only**: N=14 identical structures across ALL 9 configurations ⭐ **Recommended for rebuttal**

---

## 🎯 For Rebuttal Submission

### **Primary File (Fair Comparison)**
- **`SHARED_STRUCTURES_TABLE.txt`** (5.6 KB)
  - **Use this for your rebuttal!**
  - Fair comparison with fixed baseline (N=14 identical structures)
  - Clean table format ready for paper
  - Statistical significance tests included
  - Key findings and recommendations

### **Supporting Documentation**
- **`SHARED_STRUCTURES_ANALYSIS.txt`** (15 KB)
  - Detailed analysis of shared structures
  - Pareto frontier analysis
  - Consistency metrics (coefficient of variation)
  - Comparison to all-structures analysis

---

## 📊 Complete Analysis Files

### **Quick Reference**
- **`QUICK_REFERENCE.txt`** (8.1 KB)
  - Compact summary with both analyses (shared + all structures)
  - ASCII Pareto frontier plot
  - Quick recommendations by use case
  - Comparison table showing consistency

### **Detailed Analysis**
- **`WEIGHT_SENSITIVITY_TABLE.txt`** (6.9 KB)
  - Complete results table (all available structures)
  - Key observations and statistical significance
  - Addressing reviewer concerns
  - Conclusion and recommendations

- **`WEIGHT_SENSITIVITY_SUMMARY.txt`** (6.8 KB)
  - Original comprehensive summary (all structures)
  - Detailed findings and observations
  - Pareto front analysis
  - Recommendations by use case

- **`WEIGHT_SENSITIVITY_KEY_FINDINGS.md`** (8.9 KB)
  - Markdown format with detailed analysis
  - 5 key findings with evidence
  - Pareto front visualization (conceptual)
  - Robustness analysis and recommendations

### **Raw Data**
- **`shared_structures_analysis.json`**
  - Machine-readable statistics for shared structures
  - List of 14 shared structure IDs
  - Complete statistics for all 9 configurations

---

## 🔬 Analysis Details

### Shared Structures (N=14)
**Structure IDs**: 7dz2_C, 7eoz_A, 7fac_A, 7fgb_A, 7fgp_A, 7fh0_B, 7n3y_A, 7n6h_A, 7n99_A, 7oj1_A, 7oj2_A, 7oju_A, 7pc1_A, 7pce_A

**Why use shared structures?**
- ✅ Fair comparison: All 9 configurations tested on IDENTICAL structures
- ✅ Fixed baseline: Eliminates structure selection bias
- ✅ Statistical rigor: Paired comparisons on same test set
- ✅ Reviewer-friendly: Addresses "fair comparison" concerns

### Weight Configurations Tested (9 total)

| Configuration | w_AAR | w_scTM | w_Bio | Purpose |
|--------------|-------|--------|-------|---------|
| Sequence Dominant | 0.80 | 0.15 | 0.05 | Max sequence recovery |
| Sequence Focused | 0.70 | 0.25 | 0.05 | High sequence priority |
| **Baseline** | **0.60** | **0.35** | **0.05** | **Original config** |
| Equal Balance | 0.48 | 0.48 | 0.05 | Balanced multi-objective |
| Structure Slight Edge | 0.40 | 0.55 | 0.05 | Slight structure priority |
| Structure Focused | 0.25 | 0.70 | 0.05 | High structure priority |
| Structure Dominant | 0.15 | 0.80 | 0.05 | Max structural similarity |
| Biophysical Aware | 0.50 | 0.35 | 0.15 | Moderate bio constraints |
| Biophysical Strict | 0.45 | 0.30 | 0.25 | Strong bio constraints |

---

## 📈 Key Results Summary

### Shared Structures Analysis (N=14)

**Best Performers:**
- **Best AAR**: Biophysical Strict (+0.0241)
- **Best scTM**: Structure Slight Edge (+0.1740) — **3.7x better than sequence-focused!**
- **Best Reward**: Structure Dominant (+0.0970)
- **Best Overall**: Equal Balance (+0.001 AAR, +0.128 scTM) — balanced improvements

**Pareto Trade-off:**
- High w_AAR (0.80) → AAR +0.022, scTM +0.047
- High w_scTM (0.40) → AAR -0.026, scTM +0.174
- **Trade-off ratio**: 3.7x better scTM costs AAR decline

**Statistical Significance:**
- Sequence Dominant vs Structure Dominant: p < 0.001 (highly significant)
- Equal Balance vs Baseline: p < 0.05 (significant)
- All major trade-offs are statistically significant

---

## 💡 Key Messages for Reviewers

### Q: "Is the composite reward a tuned secret sauce?"
**A**: **NO.** The sensitivity analysis demonstrates:
- ✅ Predictable behavior: Strong correlations (r=+0.85 for AAR, r=-0.78 for scTM)
- ✅ No magic configuration: Multiple configurations perform well
- ✅ Transparent trade-offs: Clear Pareto frontier between AAR and scTM
- ✅ User control: Weights can be tuned for specific design objectives

### Q: "How sensitive is the method to weight choices?"
**A**: **Appropriately sensitive.** The method:
- ✅ Responds predictably to weight changes
- ✅ Maintains positive improvements across most configurations
- ✅ No catastrophic failures from reasonable weight choices
- ✅ Allows users to prioritize objectives based on application needs

### Q: "What are the per-metric outcomes?"
**A**: **Fully characterized.** We provide:
- ✅ Complete AAR, scTM, and reward statistics for all 9 configurations
- ✅ Clear Pareto trade-offs documented with statistical tests
- ✅ Balanced weights achieve best overall performance
- ✅ Biophysical constraints can be incorporated effectively

---

## 🎯 Recommendations

### For Different Applications

| Application | Recommended Config | Rationale |
|------------|-------------------|-----------|
| **Drug Design** | Sequence Dominant (0.80/0.15/0.05) | Prioritize sequence recovery for binding sites |
| **Structural Biology** | Structure Slight Edge (0.40/0.55/0.05) | Maximize structural similarity (+0.174 scTM) |
| **General Engineering** | Equal Balance (0.48/0.48/0.05) | Best multi-objective performance |
| **Stability-Critical** | Biophysical Aware (0.50/0.35/0.15) | Incorporate biophysical constraints |
| **Default/Baseline** | Baseline (0.60/0.35/0.05) | Original paper config, proven performance |

### General Guidance
- **Sequence-critical applications** → High w_AAR (0.70-0.80)
- **Structure-critical applications** → High w_scTM (0.55-0.80)
- **Balanced objectives** → Equal weights (0.48/0.48) or Baseline (0.60/0.35)
- **Constrained design** → Increase w_Biophysical (0.15-0.25)

---

## 🔍 Consistency Verification

### Comparison: Shared vs All Structures

| Configuration | Shared (N=14) | All (N=15-25) | Consistent? |
|--------------|---------------|---------------|-------------|
| Sequence Dominant | AAR +0.022, scTM +0.047 | AAR +0.018, scTM +0.045 | ✓ Yes |
| Baseline | AAR +0.017, scTM +0.074 | AAR +0.011, scTM +0.056 | ✓ Yes |
| Equal Balance | AAR +0.001, scTM +0.128 | AAR +0.014, scTM +0.117 | ✓ Yes |
| Structure Slight Edge | AAR -0.026, scTM +0.174 | AAR -0.018, scTM +0.179 | ✓ Yes |
| Structure Dominant | AAR -0.036, scTM +0.129 | AAR -0.043, scTM +0.142 | ✓ Yes |

**Conclusion**: Trends are CONSISTENT across both analyses, validating the findings.

---

## 📝 How to Use These Files

### For Paper/Rebuttal
1. **Use `SHARED_STRUCTURES_TABLE.txt`** for the main results table
2. **Reference `SHARED_STRUCTURES_ANALYSIS.txt`** for detailed discussion
3. **Cite `QUICK_REFERENCE.txt`** for quick comparisons

### For Presentations
1. **Use `WEIGHT_SENSITIVITY_KEY_FINDINGS.md`** for slide content
2. **Extract Pareto frontier** from QUICK_REFERENCE.txt
3. **Show consistency** with shared vs all structures comparison

### For Further Analysis
1. **Load `shared_structures_analysis.json`** for custom plots
2. **Extract specific metrics** for targeted comparisons
3. **Verify statistics** with raw data

---

## ✅ Bottom Line

The composite reward is a **transparent, tunable multi-objective optimization tool**, NOT a "secret sauce":

- ✅ Responds predictably to weight changes (strong correlations)
- ✅ Multiple configurations achieve good performance (robust)
- ✅ Clear Pareto trade-offs between AAR and scTM (statistically significant)
- ✅ Users can select weights based on specific design objectives
- ✅ Baseline (0.60/0.35/0.05) is reasonable but NOT uniquely optimal

**Recommendation**: Use **Equal Balance (0.48/0.48/0.05)** for general protein engineering as it achieves the best multi-objective performance on shared structures.

---

## 📧 Questions?

For questions about the analysis or interpretation, refer to:
- **Detailed methodology**: `SHARED_STRUCTURES_ANALYSIS.txt`
- **Quick answers**: `QUICK_REFERENCE.txt`
- **Statistical details**: `WEIGHT_SENSITIVITY_TABLE.txt`
