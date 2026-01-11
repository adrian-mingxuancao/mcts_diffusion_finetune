# DNA Aptamer 6922 Structure Prediction & Docking Results (Chai-1)

## Task: All-atom tertiary structure prediction and predicted docking of a known 80-nucleotide DNA aptamer to small proteins of known structure

**Status: ✅ WORKING**
**Backend: Chai-1 (diffusion-based, Python API)**

## Summary

This pipeline successfully predicted:
1. **Aptamer-only tertiary structures** for two 80-nt DNA aptamers (6922 and 6927)
2. **Protein-aptamer complex structures** (docking) for aptamer 6922 with three protein targets

## Results Table

| Task | Target | pLDDT | Runtime | Interpretation |
|------|--------|-------|---------|----------------|
| Aptamer 6922 only | - | 56.13 | 80s | Moderate confidence |
| Aptamer 6927 only | - | 55.25 | 73s | Moderate confidence |
| **6922 + TIRR (6D0L)** | 6D0L | **68.67** | 305s | **Good confidence** - suggests binding |
| **6922 + Nudt16TI (6CO2)** | 6CO2 | **66.92** | 106s | **Good confidence** - suggests binding |
| 6922 + Nudt16 (3COU) | 3COU | 67.83 | 105s | Good confidence |

## Comparison with Boltz

| Task | Boltz pLDDT | Chai-1 pLDDT | Difference |
|------|-------------|--------------|------------|
| Aptamer 6922 only | 40.58 | 56.13 | **+15.55** (Chai better) |
| Aptamer 6927 only | 48.05 | 55.25 | **+7.20** (Chai better) |
| 6922 + TIRR (6D0L) | 65.22 | 68.67 | **+3.45** (Chai better) |
| 6922 + Nudt16TI (6CO2) | 64.20 | 66.92 | **+2.72** (Chai better) |
| 6922 + Nudt16 (3COU) | 62.01 | 67.83 | **+5.82** (Chai better) |

**Key Finding**: Chai-1 consistently produces higher pLDDT scores than Boltz for DNA aptamer predictions.

## Method

- **Structure Prediction**: Chai-1 (diffusion-based model via Python API)
- **Diffusion Steps**: 200
- **Trunk Recycles**: 3
- **GPU**: NVIDIA A100 80GB

## Date

Generated: January 11, 2026
