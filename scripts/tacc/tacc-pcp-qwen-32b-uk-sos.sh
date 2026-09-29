#!/bin/bash

#SBATCH -J 32uksos  # Job name
#SBATCH -o ./%j.stdout.txt  # Name of stdout output file(%j expands to jobId)
#SBATCH -e ./%j.stderr.txt  # Name of stderr output file(%j expands to jobId)
#SBATCH -p gh
#SBATCH -N 1                    # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 10:00:00
#SBATCH --mail-user=marinov@utexas.edu
#SBATCH --mail-type=ALL

# Prepare modules
echo "START: $(date)"

module load gcc/13 cuda/12.4 nvidia_math/12.4 nccl/12.4
unset PYTHONPATH
module load python3
source activate llm-interpreter
echo $CONDA_DEFAULT_ENV
unset TRANSFORMERS_CACHE
export HF_HOME=$SCRATCH

cd /work/11826/ldm3484/vista/PLSemanticsBench-internal

python src/main.py experiment qwen_coder_32b_uk_pcp_cot_imp_sos --seed 42
python src/main.py experiment qwen_coder_32b_uk_pcp_cot_imp_sos --seed 73
python src/main.py experiment qwen_coder_32b_uk_pcp_cot_imp_sos --seed 94
