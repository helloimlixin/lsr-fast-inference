from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoConfig
)
from datasets import load_dataset

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
        "meta-llama/Llama-3.2-1B",
        config=config
    )
    
    # Load wikitext dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    dataset = dataset.map(lambda x: {"text": x["text"]}, batched=True)
    
    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/calibration",
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.001,
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
    )
    
    # Run calibration
    trainer.train()
    
    # Save calibrated factors
    model.save_kronecker_factors("./calibrated_factors.bin")

if __name__ == "__main__":
    calibrate_kronecker()