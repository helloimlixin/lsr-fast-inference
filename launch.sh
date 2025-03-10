# # #!/bin/bash
# # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# # # Launch training using Python's multiprocessing module (works with Hydra)
# # torchrun \
# #   --nproc_per_node=$NUM_GPUS \
# #   run.py \
# #   model=kronecker_full \
# #   training=multi_gpu

# #!/bin/bash

# # Get number of GPUs and CPU cores
# NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
# NUM_CORES=$(nproc)

# # Calculate optimal threads per GPU process
# # Reserve 2 cores for system operations
AVAILABLE_CORES=$((NUM_CORES - 2))
THREADS_PER_GPU=$((AVAILABLE_CORES / NUM_GPUS))

# Ensure at least 1 thread per process
if [ $THREADS_PER_GPU -lt 1 ]; then
    THREADS_PER_GPU=1
fi

# # Set thread-related environment variables
export OMP_NUM_THREADS=$THREADS_PER_GPU
# export MKL_NUM_THREADS=$THREADS_PER_GPU
# export NUMEXPR_NUM_THREADS=$THREADS_PER_GPU
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# echo "Running with $NUM_GPUS GPUs, $THREADS_PER_GPU threads per GPU process"

# # Launch with torch distributed
# torchrun \
#   --nproc_per_node=$NUM_GPUS \
#   run.py \
#   model=kronecker_full \
#   training=multi_gpu

#!/bin/bash

# Common defaults
MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_DIR="./outputs"
CONFIG_DIR="./config"

# Parse arguments
MODE=${1:-"train"}  # train or calibrate

case $MODE in
  calibrate)
    echo "Launching Kronecker calibration..."
    torchrun --nproc_per_node=$NUM_GPUS calibrate_kronecker.py \
      model=kronecker_calibration \
      training=calibration \
      model.kronecker.calibration=true \
      model.lora.enabled=false \
      training.output_dir="$OUTPUT_DIR/kronecker_calibration"
    ;;

  train)
    echo "Launching standard training..."
    torchrun --nproc_per_node=$NUM_GPUS run.py \
      model=kronecker_full \
      training=multi_gpu \
      model.kronecker.strategy=full \
      model.lora.enabled=true \
      training.output_dir="$OUTPUT_DIR/kronecker_lora_finetune" \
      training.dataset_name=glue \
      training.dataset_subset=mrpc
    ;;

  *)
    echo "Usage: $0 [train|calibrate] [num_gpus]"
    exit 1
    ;;
esac

# Set optimal threading
export OMP_NUM_THREADS=$(( $(nproc) / $NUM_GPUS ))
