#!/bin/bash

set -euo pipefail

num_gpus=$(nvidia-smi --list-gpus | wc -l)
num_cores=$(nproc)
available_cores=$(( num_cores > 2 ? num_cores - 2 : 1 ))
threads_per_gpu=$(( available_cores / num_gpus ))

if [ "$threads_per_gpu" -lt 1 ]; then
    threads_per_gpu=1
fi

export OMP_NUM_THREADS="$threads_per_gpu"
export MKL_NUM_THREADS="$threads_per_gpu"
export NUMEXPR_NUM_THREADS="$threads_per_gpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mode=${1:-train}
task=${2:-mrpc}
output_dir=./outputs
config_dir=./config
python_bin=${PYTHON_BIN:-python3}

run_distributed() {
    "$python_bin" -m torch.distributed.run --nproc_per_node="$num_gpus" "$@"
}

case "$mode" in
    calibrate)
        echo "Launching Kronecker calibration on WikiText..."
        run_distributed calibrate_kronecker.py
        ;;

    benchmark)
        echo "Launching dense vs Kronecker benchmark..."
        "$python_bin" benchmark_kronecker.py \
            --model-name=meta-llama/Llama-3.2-1B \
            --kronecker-implementation=gemm \
            --target-modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
        ;;

    train)
        echo "Launching standard training..."
        run_distributed run.py model=kronecker_full training=multi_gpu dataset=mrpc
        ;;

    finetune)
        if [ ! -f "$config_dir/dataset/$task.yaml" ]; then
            echo "Missing dataset config: $config_dir/dataset/$task.yaml"
            exit 1
        fi

        echo "Launching finetuning on $task..."
        run_distributed run.py \
            model=kronecker_full \
            training=multi_gpu \
            dataset="$task" \
            training.output_dir="$output_dir/finetune_$task"
        ;;

    *)
        echo "Usage: $0 [train|calibrate|benchmark|finetune] [dataset_config]"
        echo "  train: Run distributed training with the existing MRPC config"
        echo "  calibrate: Run distributed Kronecker calibration"
        echo "  benchmark: Compare dense and Kronecker LLM inference/eval runs"
        echo "  finetune: Run distributed fine-tuning for an existing dataset config"
        exit 1
        ;;
esac

echo "Done!"
