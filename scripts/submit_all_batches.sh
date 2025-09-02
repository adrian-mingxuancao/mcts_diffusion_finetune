#!/bin/bash

# Submit all CAMEO MCTS ablation batches in parallel
echo "🚀 Submitting 5 parallel CAMEO MCTS ablation batches..."

cd /home/caom/AID3/dplm/mcts_diffusion_finetune/scripts

# Submit all batches
JOB1=$(sbatch submit_cameo_batch_1.sbatch | awk '{print $4}')
JOB2=$(sbatch submit_cameo_batch_2.sbatch | awk '{print $4}')
JOB3=$(sbatch submit_cameo_batch_3.sbatch | awk '{print $4}')
JOB4=$(sbatch submit_cameo_batch_4.sbatch | awk '{print $4}')
JOB5=$(sbatch submit_cameo_batch_5.sbatch | awk '{print $4}')

echo "📊 Submitted jobs:"
echo "  Batch 1 (structures 0-3):  Job ID $JOB1"
echo "  Batch 2 (structures 4-7):  Job ID $JOB2"
echo "  Batch 3 (structures 8-11): Job ID $JOB3"
echo "  Batch 4 (structures 12-15): Job ID $JOB4"
echo "  Batch 5 (structures 16-17): Job ID $JOB5"

echo ""
echo "🔍 Monitor progress with:"
echo "  squeue -u \$USER"
echo "  tail -f /net/scratch/caom/cameo_batch*_*.out"

echo ""
echo "📁 Results will be saved to:"
echo "  /net/scratch/caom/cameo_evaluation_results/"

echo ""
echo "⏱️  Expected completion: ~8 hours per batch (parallel execution)"
echo "🎯 Total structures: 17 (4 ablation tests each = 68 total tests)"
