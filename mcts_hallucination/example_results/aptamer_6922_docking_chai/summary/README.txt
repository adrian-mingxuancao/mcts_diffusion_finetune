ABCFold DNA Aptamer Structure Prediction Results
================================================

Date: Sun Jan 11 11:43:49 AM CST 2026
Job ID: 628763

Scientific Context:
- Aptamer 6922 is a known binder to TIRR protein
- Aptamer 6927 is a non-binder control
- TIRR (6D0L) blocks 53BP1 recruitment to DNA damage sites
- Nudt16TI (6CO2) shares domain with TIRR
- Nudt16 (3COU) is parent protein (no binding expected)

Tasks Completed:
1. Aptamer-only folding: 6922 and 6927 tertiary structure prediction
2. Complex prediction: aptamer 6922 + TIRR (6D0L)
3. Complex prediction: aptamer 6922 + Nudt16TI (6CO2)
4. Complex prediction: aptamer 6922 + Nudt16 (3COU)

Expected Results:
- 6922 + TIRR/Nudt16TI: High confidence (good binding)
- 6922 + Nudt16: Low confidence (no binding)

Directory Structure:
- inputs/: FASTA files and downloaded PDBs
- aptamer_only/: Single-chain aptamer structure predictions
- complex/: Protein-aptamer complex predictions
- summary/: Metrics CSV and this README
- logs/: Job output logs

Metrics File: summary/metrics.csv
- task_name: Description of the prediction task
- target: Protein PDB ID (or "none" for aptamer-only)
- top_model_confidence: Mean pLDDT score (0-100)
- runtime_seconds: Wall-clock time for prediction
- notes: Success/failure status

TODO (Future Work):
- Aptamer sequence optimization for improved binding
- Interface analysis and contact mapping
- Comparison with experimental binding data
