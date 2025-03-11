from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoConfig,
    DataCollatorWithPadding
)
import evaluate
from datasets import load_dataset
from src.models.model_utils import replace_all_linear_with_kronecker, replace_linear_with_lora
import torch
from torch import nn
import json
import os

class CausalLMForSequenceClassification(nn.Module):
    def __init__(self, causal_model, num_labels):
        super().__init__()
        self.causal_model = causal_model  # The pretrained causal LM
        self.config = causal_model.config  # Copy the config for the classification head
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the causal LM to obtain hidden states.
        # Note: Access the underlying transformer (e.g., `model` attribute) if needed.
        outputs = self.causal_model.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Pooling strategy: use the representation of the first token (or use mean pooling)
        pooled_output = hidden_states[:, 0, :]
        
        # Compute logits using the classification head
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return {"loss": loss, "logits": logits}

def finetune_mrpc():
    # Load the calibrated model
    config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
    config.num_labels = 2
    config.update({
        "kronecker": {
            "enabled": True,
            "calibration": True,
            "max_candidate": 32,
            "als_iter": 15,
            "full_model": True
        }
    })

    # Load the base model
    model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B", config=config)

    # Apply Kronecker factorization
    model = replace_all_linear_with_kronecker(
        model,
        max_candidate=32,
        als_iter=15
    )

    # Load the calibrated weights
    model = model.from_pretrained("outputs/calibration/checkpoint-650")

    # model = CausalLMForSequenceClassification(model, num_labels=2)

    # replace_linear_with_lora(model, r=8, alpha=32)


    # Load MRPC dataset
    dataset = load_dataset("glue", "mrpc")
    
    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # Set the pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        metric = evaluate.load("glue", "mrpc")
        return metric.compute(predictions=predictions, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/mrpc_finetuned",
        eval_strategy="steps",
        eval_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=100,
        weight_decay=0.01,
        fp16=True,
        deepspeed="ds_config.json"
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Fine-tune the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    return eval_results

if __name__ == "__main__":
    eval_results = finetune_mrpc()
    print("Evaluation Results:", eval_results)