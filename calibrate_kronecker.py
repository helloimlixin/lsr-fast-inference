import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from src.models.model_utils import replace_all_linear_with_kronecker
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calibrate_kronecker(model_name="meta-llama/Llama-3.2-1B", dataset_name="wikitext", 
                        dataset_config="wikitext-2-raw-v1"):
    # Load custom config
    config = AutoConfig.from_pretrained(model_name)
    config.update({
        "kronecker": {
            "enabled": True,
            "calibration": True,
            "max_candidate": 32,
            "als_iter": 15,
        }
    })

    # Load model with updated config
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)

    # Replace linear layers with Kronecker approximation
    model = replace_all_linear_with_kronecker(model, max_candidate=32, als_iter=15)

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config)

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Group texts
    block_size = 32
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(group_texts, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/kronecker_calibration",
        eval_strategy="steps",
        eval_steps=10,
        logging_steps=10,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,
        ddp_find_unused_parameters=False,
        save_steps=100,
        save_total_limit=2,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        data_collator=data_collator,
    )

    # Run calibration
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()

    return eval_results, model, tokenizer

def generate_text(model, tokenizer, prompt, num_return_sequences=3, max_length=256):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Move input to the same device as the model
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            num_return_sequences=num_return_sequences,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_beams=5
        )

    generated_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
    return generated_texts
if __name__ == "__main__":
    eval_results, model, tokenizer = calibrate_kronecker()
    logger.info(f"Evaluation results: {eval_results}")

    eval_loss = eval_results.get("eval_loss", float("inf"))
    if math.isnan(eval_loss):
        raise ValueError("Evaluation loss is NaN")
    else:
        logger.info(f"Evaluation loss: {eval_loss}")
        perplexity = math.exp(eval_loss)
        logger.info(f"Perplexity: {perplexity}")

    model.save_pretrained("./calibrated_kronecker_model")
    tokenizer.save_pretrained("./calibrated_kronecker_model")

    # Generate text
    prompt = "Once upon a time"
    generated_texts = generate_text(model, tokenizer, prompt)
    for i, text in enumerate(generated_texts):
        logger.info(f"Generated text {i+1}: {text}")

    logger.info("Calibration complete.")