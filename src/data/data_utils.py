from datasets import load_dataset
from transformers import DataCollatorWithPadding


TEXT_PAIR_CANDIDATES = (
    ("sentence1", "sentence2"),
    ("premise", "hypothesis"),
    ("question1", "question2"),
    ("question", "sentence"),
    ("question", "passage"),
)


def _infer_text_columns(column_names):
    usable_columns = [
        name for name in column_names
        if name not in ("label", "labels", "idx")
    ]

    for pair in TEXT_PAIR_CANDIDATES:
        if all(name in usable_columns for name in pair):
            return pair

    if "text" in usable_columns:
        return ("text",)

    if "sentence" in usable_columns:
        return ("sentence",)

    if len(usable_columns) >= 2:
        return tuple(usable_columns[:2])

    if usable_columns:
        return (usable_columns[0],)

    raise ValueError("Could not infer text columns from dataset schema")


def prepare_dataset(dataset_name, subset, tokenizer, max_length=128):
    """Load a dataset, tokenize it, and return a padded collator."""
    print("Loading dataset: {}/{}".format(dataset_name, subset))
    dataset = load_dataset(dataset_name, subset)

    train_split = dataset["train"]
    label_column = "label" if "label" in train_split.column_names else "labels"
    if label_column not in train_split.column_names:
        raise ValueError("Dataset must contain a 'label' or 'labels' column")

    text_columns = _infer_text_columns(train_split.column_names)

    def tokenize_function(examples):
        texts = [examples[column] for column in text_columns]
        return tokenizer(*texts, truncation=True, max_length=max_length)

    columns_to_remove = [
        column for column in train_split.column_names
        if column != label_column
    ]

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
    )

    if label_column != "labels":
        tokenized_datasets = tokenized_datasets.rename_column(label_column, "labels")

    tokenized_datasets.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return tokenized_datasets, data_collator
