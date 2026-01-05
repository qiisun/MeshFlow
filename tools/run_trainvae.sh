CONFIG_PATH=$1

TARGET_GPU_ID=5,6
GPUS_PER_NODE=2
PRECISION=${PRECISION:-bf16}
# GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.1.0}
MASTER_PORT=${MASTER_PORT:-1240}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    --gpu_ids $TARGET_GPU_ID \
    train_vae.py \
    --config $CONFIG_PATH


