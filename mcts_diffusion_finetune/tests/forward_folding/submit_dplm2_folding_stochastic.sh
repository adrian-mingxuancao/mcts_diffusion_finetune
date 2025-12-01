#!/bin/bash
# Submit stochastic folding generation for all DPLM-2 models

echo "🚀 Submitting DPLM-2 Stochastic Folding Generation"
echo "📅 $(date)"
echo ""
echo "📊 Models to generate:"
echo "   1. DPLM-2 150M"
echo "   2. DPLM-2 650M"
echo "   3. DPLM-2 3B"
echo ""
echo "🎯 Task: Forward Folding (sequence → structure)"
echo "📈 Sampling: Stochastic (matching MCTS config)"
echo "   • Temperature: 1.0"
echo "   • Unmasking: stochastic1.0"
echo "   • Sampling: annealing@2.2:1.0"
echo "   • Max iterations: 150"
echo "📏 Dataset: CAMEO2022 (183 sequences)"
echo ""

# Submit the array job
JOB_ID=$(sbatch --parsable generate_dplm2_folding_stochastic.sh)

if [ $? -eq 0 ]; then
    echo "✅ Job submitted successfully!"
    echo "📋 Job ID: ${JOB_ID}"
    echo ""
    echo "📊 Monitor progress:"
    echo "   squeue -u \$USER"
    echo "   tail -f /home/caom/AID3/dplm/logs/dplm2_folding_stochastic_${JOB_ID}_*.out"
    echo ""
    echo "📁 Results will be saved to:"
    echo "   /home/caom/AID3/dplm/generation-results/dplm2_150m_stochastic/folding/"
    echo "   /home/caom/AID3/dplm/generation-results/dplm2_650m_stochastic/folding/"
    echo "   /home/caom/AID3/dplm/generation-results/dplm2_3b_stochastic/folding/"
else
    echo "❌ Job submission failed!"
    exit 1
fi
