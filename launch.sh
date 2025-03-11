#!/bin/bash

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# Get number of CPU cores
NUM_CORES=$(nproc)

# Calculate optimal threads per GPU process
# Reserve 2 cores for system operations
AVAILABLE_CORES=$((NUM_CORES - 2))
THREADS_PER_GPU=$((AVAILABLE_CORES / NUM_GPUS))

# Ensure at least 1 thread per process
if [ $THREADS_PER_GPU -lt 1 ]; then
    THREADS_PER_GPU=1
fi

# Set thread-related environment variables
export OMP_NUM_THREADS=$THREADS_PER_GPU
export MKL_NUM_THREADS=$THREADS_PER_GPU
export NUMEXPR_NUM_THREADS=$THREADS_PER_GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Common defaults
MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_DIR="./outputs"
CONFIG_DIR="./config"

# Parse arguments
MODE=${1:-"train"}  # train, calibrate or finetune
TASK=${2:-"mrpc"}   # For finetuning: mrpc, sst2, etc.

case $MODE in
  calibrate)
    echo "Launching Kronecker calibration on WikiText..."
    torchrun --nproc_per_node=$NUM_GPUS calibrate_kronecker.py \
      --model_name=$MODEL_NAME \
      --dataset_name="wikitext" \
      --dataset_config="wikitext-2-raw-v1"
    ;;

  train)
    echo "Launching standard training..."
    torchrun --nproc_per_node=$NUM_GPUS run.py \
      model=kronecker_full \
      training=multi_gpu
    ;;

  finetune)
    echo "Launching finetuning on GLUE $TASK..."
    torchrun --nproc_per_node=$NUM_GPUS run.py \
      model=kronecker_full \
      training=finetuning \
      model.kronecker.use_calibrated=true \
      model.kronecker.calibration_path="$OUTPUT_DIR/kronecker_calibration" \
      dataset=glue_$TASK \
      training.output_dir="$OUTPUT_DIR/finetune_$TASK"
    ;;

  *)
    echo "Usage: $0 [train|calibrate|finetune] [task_name]"
    echo "  - train: Run standard training"
    echo "  - calibrate: Run Kronecker calibration on WikiText"
    echo "  - finetune: Finetune on GLUE task (default: mrpc)"
    exit 1
    ;;
esac

echo "Done!"
