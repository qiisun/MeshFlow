#!/bin/bash
# ============================================================
# TACC Vista SLURM batch script — multi-node training
# Each Vista `gh` node has 1 GH200, so N nodes = N GPUs.
# ============================================================


#SBATCH -J meshflow-multi  # Job's name
#SBATCH -o /work/11264/guandao/vista/MeshFlow2/logs/__JOB_NAME__.%j.out # Output
#SBATCH -e /work/11264/guandao/vista/MeshFlow2/logs/__JOB_NAME__.%j.err # Error's output
#SBATCH -p gh              # node name
#SBATCH -N __NUM_NODES__   # how many nodes
#SBATCH -n __NUM_NODES__   # one task per node (one GPU per node on gh)
#SBATCH -t 48:00:00        # Duration
#SBATCH -A ASC26034        # Project ID
source ~/.bashrc
module load gcc/13 cuda/12.4 nccl/12.4 nvidia_math
export CC=gcc
export CXX=g++
export TORCH_CUDA_ARCH_LIST="9.0a"
conda activate meshflow
cd $HOME/MeshFlow2

# Rendezvous configuration — pick the first allocated node as master.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NUM_NODES=$SLURM_NNODES
export NUM_PROCESSES=$SLURM_NNODES   # 1 GPU per node on Vista gh

# NCCL tuning for Vista's interconnect.
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=^lo,docker

echo "Python        : $(which python)"
echo "Conda env     : $CONDA_DEFAULT_ENV"
echo "Nodes         : $SLURM_JOB_NODELIST"
echo "Master        : $MASTER_ADDR:$MASTER_PORT"
echo "Num processes : $NUM_PROCESSES"

# Expand parent-shell vars (NUM_NODES/PROCESSES, MASTER_ADDR/PORT) up-front via
# double quotes; defer only $SLURM_NODEID with a backslash since it differs per
# task. Single-quoted bash -c was unreliable on Vista — env vars sometimes
# failed to propagate into the srun subshell, leaving accelerate to fall back
# to single-process mode (each node ran standalone with world_size=1).
srun --nodes=$NUM_NODES --ntasks-per-node=1 --export=ALL bash -c "
    echo \"[task \$SLURM_NODEID] accelerate launch --num_machines $NUM_NODES --num_processes $NUM_PROCESSES --machine_rank \$SLURM_NODEID --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT\"
    accelerate launch \
        --num_machines $NUM_NODES \
        --num_processes $NUM_PROCESSES \
        --machine_rank \$SLURM_NODEID \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --mixed_precision bf16 \
        train.py --config __CONFIG_PLACEHOLDER__ __OVERRIDES_PLACEHOLDER__
"
echo "Job finished at $(date)"
