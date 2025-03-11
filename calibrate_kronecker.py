from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoConfig
)
from datasets import load_dataset
from src.models.model_utils import replace_all_linear_with_kronecker, replace_linear_with_lora
import math

def calibrate_kronecker():
    # Load custom config first
    config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
    config.update({
        "kronecker": {
            "enabled": True,
            "calibration": True,
            "calibration_dataset": "wikitext",
            "calibration_steps": 500,
            "max_candidate": 32,
            "als_iter": 15,
            "full_model": False
        }
    })
    
    # Load model with updated config
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B"
    )

    model = replace_all_linear_with_kronecker(
                model,
                max_candidate=32,
                als_iter=15
            )
    
    # Load wikitext dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    dataset = dataset.map(lambda x: {"text": x["text"]}, batched=True)
    
    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Group texts into blocks
    block_size = 64
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(group_texts, batched=True)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/calibration",
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        num_train_epochs=50,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        weight_decay=0.1,
        fp16=True,
        deepspeed="ds_config.json"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        data_collator=data_collator,
    )
    
    # Run calibration
    trainer.train(resume_from_checkpoint=True)

    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss", 0.0)
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    return eval_results, perplexity
    
if __name__ == "__main__":
    eval_results, perplexity = calibrate_kronecker()

    print("Evaluation Results:", eval_results)
    print("Perplexity:", perplexity)