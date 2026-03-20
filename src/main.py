from pathlib import Path

from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import prepare_dataset
from src.models import LoRALinear
from src.models.model_utils import (
    get_kronecker_stats,
    replace_all_linear_with_kronecker,
    replace_classifier_with_kronecker,
    replace_linear_with_lora,
)
from src.training import setup_trainer


CLASSIFIER_ATTRS = ("classifier", "score")


def _get_classifier_attr(model):
    for attr_name in CLASSIFIER_ATTRS:
        if hasattr(model, attr_name):
            return attr_name
    return None


def _configure_tokenizer_padding(model, tokenizer):
    if tokenizer.pad_token is not None:
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return

    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id


def apply_model_compression(model, cfg: DictConfig):
    """Apply the configured Kronecker and LoRA transforms."""
    stats = {}
    kronecker_cfg = cfg.model.kronecker
    kronecker_kwargs = {
        "max_candidate": kronecker_cfg.max_candidate,
        "als_iter": kronecker_cfg.als_iter,
        "factorization_objective": kronecker_cfg.get("factorization_objective", "balanced"),
        "tile_multiple": kronecker_cfg.get("tile_multiple", 16),
        "min_factor_size": kronecker_cfg.get("min_factor_size", 32),
        "implementation": kronecker_cfg.get("implementation", "gemm"),
    }

    if kronecker_cfg.enabled:
        strategy = kronecker_cfg.get("strategy", "classifier")

        if strategy == "full":
            print("Applying Kronecker approximation to all linear layers...")
            model = replace_all_linear_with_kronecker(
                model,
                target_modules=kronecker_cfg.get("target_modules"),
                **kronecker_kwargs
            )
            stats.update(get_kronecker_stats(model))
        elif strategy == "classifier":
            classifier_attr = _get_classifier_attr(model)
            if classifier_attr is None:
                print("Skipping Kronecker approximation: no classifier head found.")
            else:
                print("Applying Kronecker approximation to classifier only...")
                classifier = getattr(model, classifier_attr)
                setattr(
                    model,
                    classifier_attr,
                    replace_classifier_with_kronecker(
                        classifier,
                        **kronecker_kwargs
                    ),
                )
                stats.update(get_kronecker_stats(getattr(model, classifier_attr)))
        else:
            raise ValueError("Unsupported Kronecker strategy: {}".format(strategy))

    if cfg.model.lora.enabled:
        print("Applying LoRA adaptation...")
        replace_linear_with_lora(
            model,
            r=cfg.model.lora.rank,
            alpha=cfg.model.lora.alpha,
        )

    return model, stats


def print_lora_statistics(model):
    """Print a compact summary of all LoRA layers in the model."""
    lora_layers = [
        (name, module.get_stats())
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]

    if not lora_layers:
        return

    print("\n=== LoRA Parameter Reduction Statistics ===")

    total_original_params = 0
    total_lora_params = 0
    for name, stats in lora_layers:
        total_original_params += stats.original_params
        total_lora_params += stats.lora_params
        print(
            "{}: rank={}, saved={} params ({:.2f}% reduction)".format(
                name,
                stats.lora_rank,
                stats.params_saved,
                stats.reduction_percentage,
            )
        )

    total_saved = total_original_params - total_lora_params
    total_reduction = 100.0 * total_saved / total_original_params
    print("Overall: saved {} params ({:.2f}% reduction)".format(total_saved, total_reduction))
    print("=" * 40)


def _save_eval_results(output_dir, eval_results):
    results_path = Path(output_dir) / "results.txt"
    with results_path.open("w") as handle:
        for key, value in sorted(eval_results.items()):
            handle.write("{}: {}\n".format(key, value))


def run_experiment(cfg: DictConfig):
    """Run the configured fine-tuning experiment."""
    print("Loading model: {}".format(cfg.model.name))
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=cfg.model.num_labels,
        ignore_mismatched_sizes=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    _configure_tokenizer_padding(model, tokenizer)

    model, compression_stats = apply_model_compression(model, cfg)
    if compression_stats:
        print("\nModel Compression Statistics:")
        for key, value in sorted(compression_stats.items()):
            print("{}: {}".format(key, value))

    tokenized_datasets, data_collator = prepare_dataset(
        cfg.dataset.name,
        cfg.dataset.subset,
        tokenizer,
        cfg.dataset.max_length,
    )

    trainer = setup_trainer(
        model=model,
        tokenized_datasets=tokenized_datasets,
        data_collator=data_collator,
        training_args=cfg.training,
    )

    print_lora_statistics(model)

    print(
        "Starting fine-tuning on {}/{}...".format(
            cfg.dataset.name,
            cfg.dataset.subset,
        )
    )
    trainer.train()

    eval_results = trainer.evaluate()
    print("Evaluation Results:")
    print(eval_results)
    _save_eval_results(cfg.training.output_dir, eval_results)
    return model, eval_results
