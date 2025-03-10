from omegaconf import DictConfig
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import evaluate


def compute_metrics(eval_pred, metric_name="glue", subset="mrpc"):
    """
    Compute evaluation metrics for the model.

    Args:
        eval_pred: Tuple of model predictions and labels.
        metric_name: Name of the metric to use.
        subset: Subset of the metric (e.g., "mrpc.yaml.yaml" for GLUE).

    Returns:
        Dictionary of metric values.
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    metric = evaluate.load(metric_name, subset)
    return metric.compute(predictions=predictions, references=labels)


def setup_trainer(model, tokenized_datasets, data_collator, training_args):
    """
    Set up a trainer for fine-tuning.

    Args:
        model: Model to fine-tune.
        tokenized_datasets: Preprocessed datasets.
        data_collator: Data collator for batching.
        training_args: Training configuration.

    Returns:
        Configured Trainer.
    """
    # Convert Hydra config.yaml to TrainingArguments
    args = TrainingArguments(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        eval_strategy=training_args.eval_strategy,
        save_strategy=training_args.save_strategy,
        logging_steps=training_args.logging_steps,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        load_best_model_at_end=training_args.load_best_model_at_end,
        metric_for_best_model=training_args.metric_for_best_model,
        greater_is_better=training_args.greater_is_better,
    )

    # Set up early stopping callback
    callbacks = []
    if training_args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience
            )
        )

    # Create compute_metrics function with configured metric and subset
    def metric_fn(eval_pred):
        return compute_metrics(
            eval_pred,
            metric_name=training_args.metric_name,
            subset=training_args.metric_subset
        )

    # Set up trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=metric_fn,
        callbacks=callbacks,
    )

    return trainer