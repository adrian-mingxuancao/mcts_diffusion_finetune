# DNA Aptamer 6922 Structure Prediction & Docking Results (Boltz)

## Task: All-atom tertiary structure prediction and predicted docking of a known 80-nucleotide DNA aptamer to small proteins of known structure

**Status: ✅ WORKING**
**Backend: Boltz (diffusion-based)**

## Summary

This pipeline successfully predicted:
1. **Aptamer-only tertiary structures** for two 80-nt DNA aptamers (6922 and 6927)
2. **Protein-aptamer complex structures** (docking) for aptamer 6922 with three protein targets

## Results Table

| Task | Target | pLDDT | Runtime | Interpretation |
|------|--------|-------|---------|----------------|
| Aptamer 6922 only | - | 40.58 | 49s | Low confidence - DNA structure prediction is challenging |
| Aptamer 6927 only | - | 48.05 | 49s | Low confidence - DNA structure prediction is challenging |
| **6922 + TIRR (6D0L)** | 6D0L | **65.22** | 79s | **Moderate-good confidence** - suggests binding |
| **6922 + Nudt16TI (6CO2)** | 6CO2 | **64.20** | 90s | **Moderate-good confidence** - suggests binding |
| 6922 + Nudt16 (3COU) | 3COU | 62.01 | 214s | Moderate confidence |

## Interpretation

### pLDDT Score Meaning (0-100 scale)
- **>70**: High confidence - reliable structure prediction
- **60-70**: Moderate confidence - structure likely correct but some uncertainty
- **50-60**: Low confidence - significant uncertainty
- **<50**: Very low confidence - structure may be unreliable

### Key Findings

1. **Aptamer-only predictions (pLDDT ~40-48)**:
   - DNA tertiary structure prediction without a binding partner is inherently difficult
   - Low pLDDT is expected for flexible single-stranded DNA
   - The aptamer likely adopts a more defined structure upon protein binding

2. **Complex predictions (pLDDT ~62-65)**:
   - All three protein-aptamer complexes show improved confidence vs aptamer-only
   - This suggests the aptamer adopts a more stable conformation when bound
   - **TIRR (6D0L)**: Highest pLDDT (65.22) - best predicted binding
   - **Nudt16TI (6CO2)**: Similar pLDDT (64.20) - expected since it shares domain with TIRR
   - **Nudt16 (3COU)**: Slightly lower pLDDT (62.01) - may indicate weaker binding

3. **Biological Context**:
   - Aptamer 6922 is a known binder to TIRR protein
   - TIRR blocks 53BP1 recruitment to DNA damage sites
   - Nudt16TI shares a domain with TIRR, explaining similar binding prediction
   - Nudt16 is the parent protein without the TIRR-specific domain

### Limitations

1. **pLDDT differences are small** (~3 points between targets) - not conclusive for binding specificity
2. **No experimental validation** - these are computational predictions
3. **Interface quality not assessed** - would need contact analysis for binding site details

## Output Files

```
aptamer_6922_docking/
├── inputs/                    # Input FASTA files and downloaded PDBs
│   ├── aptamer_6922.fasta
│   ├── aptamer_6927.fasta
│   ├── 6D0L.pdb (TIRR)
│   ├── 6CO2.pdb (Nudt16TI)
│   └── 3COU.pdb (Nudt16)
├── aptamer_only/              # Single-chain aptamer structures
│   ├── 6922/best_model.cif
│   └── 6927/best_model.cif
├── complex/                   # Protein-aptamer complex structures
│   ├── 6922_6D0L/best_model.cif  ← TIRR complex
│   ├── 6922_6CO2/best_model.cif  ← Nudt16TI complex
│   └── 6922_3COU/best_model.cif  ← Nudt16 complex
└── summary/
    ├── metrics.csv
    └── README.txt
```

## How to View Structures

The `.cif` files can be visualized in:
- **PyMOL**: `load best_model.cif`
- **ChimeraX**: File → Open
- **Mol***: https://molstar.org/viewer/

## Method

- **Structure Prediction**: Boltz (diffusion-based model)
- **MSA Generation**: ColabFold MSA server
- **GPU**: NVIDIA A100 80GB

## Date

Generated: January 11, 2026
