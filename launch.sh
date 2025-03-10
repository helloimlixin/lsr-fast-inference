# #!/bin/bash
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # Get number of GPUs
# NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# # Launch training using Python's multiprocessing module (works with Hydra)
# torchrun \
#   --nproc_per_node=$NUM_GPUS \
#   run.py \
#   model=kronecker_full \
#   training=multi_gpu

#!/bin/bash

# Get number of GPUs and CPU cores
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
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

echo "Running with $NUM_GPUS GPUs, $THREADS_PER_GPU threads per GPU process"

# Launch with torch distributed
torchrun \
  --nproc_per_node=$NUM_GPUS \
  run.py \
  model=kronecker_full \
  training=multi_gpu