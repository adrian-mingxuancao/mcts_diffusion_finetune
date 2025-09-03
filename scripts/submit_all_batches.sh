#!/bin/bash

# Submit all CAMEO MCTS ablation jobs (method-by-method)
echo "🚀 Submitting method-by-method CAMEO MCTS ablation jobs..."

cd /home/caom/AID3/dplm/mcts_diffusion_finetune/scripts

# Submit method jobs
JOB_RANDOM=$(sbatch submit_mode_random_no_expert.sbatch | awk '{print $4}')
JOB_E0=$(sbatch submit_mode_single_expert_0.sbatch | awk '{print $4}')
JOB_E1=$(sbatch submit_mode_single_expert_1.sbatch | awk '{print $4}')
JOB_E2=$(sbatch submit_mode_single_expert_2.sbatch | awk '{print $4}')
JOB_MULTI=$(sbatch submit_mode_multi_expert.sbatch | awk '{print $4}')

echo "📊 Submitted jobs:"
echo "  random_no_expert: Job ID $JOB_RANDOM"
echo "  single_expert_0: Job ID $JOB_E0"
echo "  single_expert_1: Job ID $JOB_E1"
echo "  single_expert_2: Job ID $JOB_E2"
echo "  multi_expert:    Job ID $JOB_MULTI"

echo ""
echo "🔍 Monitor progress with:"
echo "  squeue -u \$USER"
echo "  tail -f /net/scratch/caom/cameo_mode_*_*.out"

echo ""
echo "📁 Results will be saved to:"
echo "  /net/scratch/caom/cameo_evaluation_results/"

echo ""
echo "⏱️  Each job covers ALL structures for its mode."
