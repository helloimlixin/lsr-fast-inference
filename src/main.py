import os
from omegaconf import DictConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from src.models.model_utils import (
    replace_classifier_with_kronecker,
    replace_linear_with_lora,
)
from src.data.data_utils import prepare_dataset
from src.training.trainer import setup_trainer


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
    apply_kronecker_approx = False
    if hasattr(model, "classifier") and cfg.model.kronecker.enabled:
        print("Replacing model.classifier with Kronecker approximation...")
        model.classifier = replace_classifier_with_kronecker(
            model.classifier,
            max_candidate=cfg.model.kronecker.max_candidate,
            als_iter=cfg.model.kronecker.als_iter
        )
        apply_kronecker_approx = True
    elif hasattr(model, "score") and cfg.model.kronecker.enabled:
        print("Replacing model.score with Kronecker approximation...")
        model.score = replace_classifier_with_kronecker(
            model.score,
            max_candidate=cfg.model.kronecker.max_candidate,
            als_iter=cfg.model.kronecker.als_iter
        )
        apply_kronecker_approx = True

    if not apply_kronecker_approx:
        print("No classifier head found or Kronecker replacement disabled. Skipping.")

    # Step 2: Apply LoRA adaptation to transformer layers
    if hasattr(model, "model") and cfg.model.lora.enabled:
        print("Applying LoRA adaptation to transformer linear layers...")
        replace_linear_with_lora(
            model.model,
            r=cfg.model.lora.rank,
            alpha=cfg.model.lora.alpha
        )
    else:
        print("No transformer module found or LoRA disabled. Skipping.")

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