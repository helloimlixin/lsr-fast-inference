# lsr-fast-inference
Fast LLM with Matrix Low-Separation Rank Approximation

## Project Structure

```
kronecker_lora_finetune/
├── config/
│   ├── config.yaml               # Main configuration
│   ├── model/                    # Model configurations
│   │   ├── default.yaml
│   │   └── llama.yaml
│   ├── dataset/                  # Dataset configurations
│   │   ├── default.yaml
│   │   └── mrpc.yaml
│   └── training/                 # Training configurations
│       ├── default.yaml
│       └── mrpc.yaml
├── src/
│   ├── __init__.py
│   ├── main.py                   # Entry point
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_utils.py         # Dataset loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── kronecker_linear.py   # KroneckerLinear module
│   │   ├── lora_linear.py        # LoRALinear module
│   │   └── model_utils.py        # Model modification utilities
│   └── training/
│       ├── __init__.py
│       └── trainer.py            # Training setup and evaluation
└── run.py                        # Hydra entry point
```
