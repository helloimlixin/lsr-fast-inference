import inspect

import evaluate
from omegaconf import OmegaConf
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments


TRAINING_ARGUMENT_EXTRAS = {"early_stopping_patience", "metric_name", "metric_subset"}


def compute_metrics(eval_pred, metric):
    """Compute metrics for a batch of evaluation predictions."""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def _build_training_arguments(training_cfg):
    raw_args = OmegaConf.to_container(training_cfg, resolve=True)
    supported_args = set(inspect.signature(TrainingArguments.__init__).parameters)

    eval_key = "eval_strategy"
    if "eval_strategy" not in supported_args and "evaluation_strategy" in supported_args:
        eval_key = "evaluation_strategy"

    normalized_args = {}
    for key, value in raw_args.items():
        if key in TRAINING_ARGUMENT_EXTRAS or value is None:
            continue

        target_key = eval_key if key == "eval_strategy" else key
        if target_key in supported_args:
            normalized_args[target_key] = value

    return TrainingArguments(**normalized_args)


def _get_eval_split(tokenized_datasets):
    for split_name in ("validation", "validation_matched", "test"):
        if split_name in tokenized_datasets:
            return tokenized_datasets[split_name]
    raise ValueError("No evaluation split found in tokenized datasets")


def setup_trainer(model, tokenized_datasets, data_collator, training_args):
    """Create a Trainer from the Hydra training config."""
    metric = evaluate.load(training_args.metric_name, training_args.metric_subset)
    args = _build_training_arguments(training_args)

    callbacks = []
    if training_args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience
            )
        )

    return Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=_get_eval_split(tokenized_datasets),
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, metric),
        callbacks=callbacks,
    )
