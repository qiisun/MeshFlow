#!/bin/bash
# ============================================================
# TACC Vista SLURM batch script
# ============================================================


#SBATCH -J meshflow        # Job's name
#SBATCH -o /work/11264/guandao/vista/MeshFlow2/logs/meshflow.%j.out # Output
#SBATCH -e /work/11264/guandao/vista/MeshFlow2/logs/meshflow.%j.err # Error's output
#SBATCH -p gh              # node name
#SBATCH -N 1               # how many nodes
#SBATCH -n 1               # how many tasks
#SBATCH -t 48:00:00        # Duration
#SBATCH -A ASC26034        # Project ID
source ~/.bashrc
module load gcc/13 cuda/12.4 nccl/12.4 nvidia_math
export CC=gcc
export CXX=g++
export TORCH_CUDA_ARCH_LIST="9.0a"
conda activate meshflow
cd $HOME/MeshFlow2
echo "Python        : $(which python)"
echo "Conda env     : $CONDA_DEFAULT_ENV"
accelerate launch --num_processes 1 --mixed_precision bf16 train.py --config __CONFIG_PLACEHOLDER__
echo "Job finished at $(date)"
