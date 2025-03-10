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

## Implementation Details

### Kronecker Linear

The `KroneckerLinear` module approximates a full weight matrix using the Kronecker product of two smaller matrices, significantly reducing the number of parameters while maintaining good performance.

### LoRA Linear

The `LoRALinear` module implements Low-Rank Adaptation, which keeps the original weights frozen and learns low-rank update matrices. This approach is parameter-efficient for fine-tuning large models.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.