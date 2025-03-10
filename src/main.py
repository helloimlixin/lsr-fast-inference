import os
from omegaconf import DictConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.models import LoRALinear
from src.models.model_utils import (
    replace_classifier_with_kronecker,
    replace_linear_with_lora, replace_all_linear_with_kronecker, get_kronecker_stats,
)
from src.data.data_utils import prepare_dataset
from src.training.trainer import setup_trainer

def apply_model_compression(model, cfg):
    """
    Apply model compression strategy (Kronecker and/or LoRA) based on configuration.

    Args:
        model: The model to modify.
        cfg: Configuration object.

    Returns:
        Modified model and compression statistics.
    """
    stats = {}

    if cfg.model.kronecker.enabled:
        strategy = cfg.model.kronecker.get("strategy", "classifier")

        if strategy == "full":
            print("Applying Kronecker approximation to all linear layers...")
            model = replace_all_linear_with_kronecker(
                model,
                max_candidate=cfg.model.kronecker.max_candidate,
                als_iter=cfg.model.kronecker.als_iter
            )
            stats.update(get_kronecker_stats(model))

        elif strategy == "classifier":
            print("Applying Kronecker approximation to classifier only...")
            if hasattr(model, "classifier"):
                model.classifier = replace_classifier_with_kronecker(
                    model.classifier,
                    max_candidate=cfg.model.kronecker.max_candidate,
                    als_iter=cfg.model.kronecker.als_iter
                )
            elif hasattr(model, "score"):
                model.score = replace_classifier_with_kronecker(
                    model.score,
                    max_candidate=cfg.model.kronecker.max_candidate,
                    als_iter=cfg.model.kronecker.als_iter
                )

    if cfg.model.lora.enabled:
        print("Applying LoRA adaptation...")
        replace_linear_with_lora(
            model,
            r=cfg.model.lora.rank,
            alpha=cfg.model.lora.alpha
        )

    return model, stats


def print_lora_statistics(model):
    """
    Print parameter reduction statistics for all LoRA layers in the model.

    Args:
        model: The PyTorch model containing LoRA layers
    """
    print("\n=== LoRA Parameter Reduction Statistics ===\n")

    total_original_params = 0
    total_lora_params = 0
    total_params_saved = 0

    # Find all LoRA layers
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            stats = module.get_stats()
            print(f"\nLayer: {name}")
            print(str(stats))

            total_original_params += stats.original_params
            total_lora_params += stats.lora_params
            total_params_saved += stats.params_saved

    # Print total statistics
    if total_original_params > 0:
        total_reduction = (1 - total_lora_params / total_original_params) * 100
        print("\n=== Overall Statistics ===")
        print(f"Total original parameters: {total_original_params:,}")
        print(f"Total LoRA parameters: {total_lora_params:,}")
        print(f"Total parameters saved: {total_params_saved:,}")
        print(f"Overall parameter reduction: {total_reduction:.2f}%")
    print("\n" + "=" * 40 + "\n")


def run_experiment(cfg: DictConfig):
    """
    Run the fine-tuning experiment with the given configuration.

    Args:
        cfg: Hydra configuration containing model, dataset, and training settings

    Returns:
        Trained model and evaluation results
    """
    # Load model and tokenizer
    print(f"Loading model: {cfg.model.name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=cfg.model.num_labels,
        ignore_mismatched_sizes=True
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    # Set up tokenizer padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Step 1: Apply Kronecker approximation to classifier head
    # Apply model compression strategy
    model, compression_stats = apply_model_compression(model, cfg)

    # Log compression statistics
    if compression_stats:
        print("\nModel Compression Statistics:")
        for key, value in compression_stats.items():
            print(f"{key}: {value}")


    # Prepare dataset
    tokenized_datasets, data_collator = prepare_dataset(
        cfg.dataset.name,
        cfg.dataset.subset,
        tokenizer,
        cfg.dataset.max_length
    )

    # Set up trainer
    trainer = setup_trainer(
        model=model,
        tokenized_datasets=tokenized_datasets,
        data_collator=data_collator,
        training_args=cfg.training
    )

    print_lora_statistics(model)

    # Start training
    print(f"Starting fine-tuning on {cfg.dataset.name}/{cfg.dataset.subset}...")
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    print("Evaluation Results:")
    print(eval_results)

    # Save results
    results_path = os.path.join(cfg.training.output_dir, "results.txt")
    with open(results_path, "w") as f:
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")

    return model, eval_results