from datasets import load_dataset
from transformers import DataCollatorWithPadding


def prepare_dataset(dataset_name, subset, tokenizer, max_length=128):
    """
    Load and preprocess a dataset.

    Args:
        dataset_name: Name of the dataset to load (e.g., "glue").
        subset: Subset of the dataset (e.g., "mrpc").
        tokenizer: Tokenizer for preprocessing.
        max_length: Maximum sequence length for tokenization.

    Returns:
        Tokenized datasets and data collator.
    """
    print(f"Loading dataset: {dataset_name}/{subset}")
    dataset = load_dataset(dataset_name, subset)

    def tokenize_function(examples):
        # Handle different dataset formats
        if subset == "mrpc":
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        else:
            # Default tokenization for text classification
            return tokenizer(
                examples["text"] if "text" in examples else examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

    # Tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Remove columns that are no longer needed
    if subset == "mrpc":
        columns_to_remove = ["sentence1", "sentence2", "idx"]
    else:
        columns_to_remove = ["text"] if "text" in dataset["train"].column_names else ["sentence"]
        if "idx" in dataset["train"].column_names:
            columns_to_remove.append("idx")

    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)

    # Rename label column to labels (required by Trainer)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Set format to PyTorch tensors
    tokenized_datasets.set_format("torch")

    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenized_datasets, data_collator