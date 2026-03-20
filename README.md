# Kronecker-LoRA Fine-tuning

This project implements efficient fine-tuning of large language models using a combination of Kronecker factorization for the classifier head and Low-Rank Adaptation (LoRA) for other linear layers. The code is organized into a modular structure and uses Hydra for configuration management.

## Project Structure

```
kronecker_lora_finetune/
├── config/               # Configuration files
│   ├── config.yaml       # Main configuration
│   ├── model/            # Model configurations
│   ├── dataset/          # Dataset configurations
│   └── training/         # Training configurations
├── src/                  # Source code
│   ├── data/             # Dataset utilities
│   ├── models/           # Model components
│   └── training/         # Training utilities
└── run.py                # Main entry point
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kronecker_lora_finetune.git
cd kronecker_lora_finetune

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
evaluate>=0.4.0
hydra-core>=1.3.2
omegaconf>=2.3.0
```

## Usage

### Basic Usage

To run the fine-tuning with default settings:

```bash
python run.py
```

### Customizing Configuration

You can override configuration values directly from the command line:

```bash
# Use a different model
python run.py model=default

# Change learning rate
python run.py training.learning_rate=1e-4

# Use a different dataset
python run.py dataset=default

# Disable LoRA
python run.py model.lora.enabled=false
```

### Running Multiple Experiments (Sweeps)

Hydra allows you to run parameter sweeps easily:

```bash
# Try different learning rates
python run.py -m training.learning_rate=1e-5,2e-5,3e-5

# Try different LoRA ranks
python run.py -m model.lora.rank=2,4,8,16
```

### LLM Benchmarking

To compare dense and Kronecker-approximated LLM inference on CUDA-oriented layer shapes:

```bash
python benchmark_kronecker.py \
  --model-name meta-llama/Llama-3.2-1B \
  --dtype bf16 \
  --factorization-objective latency \
  --kronecker-implementation gemm \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch-sizes 1,4 \
  --prefill-lengths 128,512 \
  --tasks wikitext103,c4_en,hellaswag,boolq,arc_easy,arc_challenge
```

To stream the full benchmark online to Weights & Biases with comparison dashboards:

```bash
python benchmark_kronecker.py \
  --model-name meta-llama/Llama-3.2-1B \
  --dtype bf16 \
  --factorization-objective latency \
  --kronecker-implementation gemm \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch-sizes 1,4 \
  --prefill-lengths 128,512 \
  --tasks wikitext103,c4_en,hellaswag,boolq,arc_easy,arc_challenge \
  --wandb-project kronecker-inference \
  --wandb-mode online
```

The benchmark script reports:

- Layer-level dense vs Kronecker latency and approximation error
- Full-model prefill and decode throughput
- Held-out quality on strong text and reasoning datasets

When Weights & Biases logging is enabled, the run also uploads:

- Raw dense, Kronecker, and comparison tables
- Layer speedup, compression-ratio, and error scatter visualizations
- Prefill/decode latency and throughput charts across context lengths
- Per-task dense vs Kronecker evaluation bar charts
- The full JSON benchmark report as a W&B artifact

You can compare the old contraction path against the new CUDA-friendlier GEMM path by switching:

```bash
python benchmark_kronecker.py --kronecker-implementation einsum
python benchmark_kronecker.py --kronecker-implementation gemm
```

Built-in evaluation tasks:

- `wikitext103`: perplexity on WikiText-103 validation
- `c4_en`: perplexity on C4 English validation
- `hellaswag`: commonsense multiple-choice accuracy
- `boolq`: boolean question answering accuracy
- `arc_easy`: science QA multiple-choice accuracy
- `arc_challenge`: harder science QA multiple-choice accuracy

## Implementation Details

### Kronecker Linear

The `KroneckerLinear` module approximates a full weight matrix using the Kronecker product of two smaller matrices, significantly reducing the number of parameters while maintaining good performance.

### LoRA Linear

The `LoRALinear` module implements Low-Rank Adaptation, which keeps the original weights frozen and learns low-rank update matrices. This approach is parameter-efficient for fine-tuning large models.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License - see the LICENSE file for details.
