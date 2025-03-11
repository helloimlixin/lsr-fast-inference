import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
import shutil
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset, Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KroneckerLinear(nn.Module):
    def __init__(self, in_features, out_features, in_factor1, in_factor2, 
                 out_factor1, out_factor2, bias=True, num_factors=4, layer_idx=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_factor1 = in_factor1
        self.in_factor2 = in_factor2
        self.out_factor1 = out_factor1
        self.out_factor2 = out_factor2
        self.num_factors = num_factors
        self.is_kronecker_layer = True
        self.layer_idx = layer_idx

        # Initialize multiple Kronecker factors
        self.K1_factors = nn.ParameterList([
            nn.Parameter(torch.randn(out_factor1, in_factor1) / np.sqrt(in_factor1))
            for _ in range(num_factors)
        ])

        self.K2_factors = nn.ParameterList([
            nn.Parameter(torch.randn(out_factor2, in_factor2) / np.sqrt(in_factor2))
            for _ in range(num_factors)
        ])

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        device = x.device

        # Move Kronecker factors to the same device as the input
        self.K1_factors = nn.ParameterList([k.to(device) for k in self.K1_factors])
        self.K2_factors = nn.ParameterList([k.to(device) for k in self.K2_factors])
        if self.bias is not None:
            self.bias = self.bias.to(device)

        original_shape = x.shape
        if x.dim() == 2:
            batch_size, seq_len = x.size(0), 1
        elif x.dim() == 3:
            batch_size, seq_len, _ = x.shape
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")

        # Reshape input to 2D: [batch_size * seq_len, in_features]
        x = x.view(-1, self.in_features)

        # Pad or truncate input if necessary
        if x.size(1) != self.in_features:
            logger.warning(f"Input size {x.size(1)} doesn't match expected {self.in_features}. Adjusting...")
            if x.size(1) < self.in_features:
                x = F.pad(x, (0, self.in_features - x.size(1)))
            else:
                x = x[:, :self.in_features]

        x_reshaped = x.view(-1, self.in_factor1, self.in_factor2)

        # Apply multiple Kronecker factors and sum results
        out = torch.zeros(batch_size * seq_len, self.out_factor1, self.out_factor2, device=device)

        for i in range(self.num_factors):
            # First multiplication
            temp = torch.matmul(x_reshaped, self.K2_factors[i].t())
            # Second multiplication
            factor_out = torch.matmul(temp, self.K1_factors[i].t())
            # Add to output
            out += factor_out

        # Reshape to original output dimensions
        out = out.view(batch_size, seq_len, self.out_features)

        # Adjust output shape if necessary
        if out.shape != original_shape:
            out = out[:original_shape[0], :original_shape[1], :original_shape[2]]

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias

        return out

    @classmethod
    def from_linear(cls, linear_layer, num_factors=4, layer_idx=None, auto_factorize=False):
        """Create a KroneckerLinear layer from a standard Linear layer"""
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        # Determine factor sizes
        if auto_factorize:
            # Find factors that divide the dimensions evenly
            in_factor1 = find_closest_factor(in_features)
            in_factor2 = in_features // in_factor1

            out_factor1 = find_closest_factor(out_features)
            out_factor2 = out_features // out_factor1
        else:
            # Use square root approximation
            in_factor1 = int(np.ceil(np.sqrt(in_features)))
            in_factor2 = int(np.ceil(in_features / in_factor1))

            out_factor1 = int(np.ceil(np.sqrt(out_features)))
            out_factor2 = int(np.ceil(out_features / out_factor1))
        # Create Kronecker layer
        kronecker_layer = cls(
            in_features=in_features,
            out_features=out_features,
            in_factor1=in_factor1,
            in_factor2=in_factor2,
            out_factor1=out_factor1,
            out_factor2=out_factor2,
            bias=linear_layer.bias is not None,
            num_factors=num_factors,
            layer_idx=layer_idx
        )

        # Move the created Kronecker layer to the same device as the original linear layer
        kronecker_layer = kronecker_layer.to(linear_layer.weight.device)

        # Copy bias if exists
        if linear_layer.bias is not None:
            kronecker_layer.bias.data.copy_(linear_layer.bias.data)

        # Initialize factors to approximate the original weight matrix
        original_weight = linear_layer.weight.data.cpu()

        # Perform SVD-based initialization for the first factor
        try:
            # Reshape for SVD
            W = original_weight.reshape(out_factor1, out_factor2, in_factor1, in_factor2)
            W = W.permute(0, 2, 1, 3).reshape(out_factor1 * in_factor1, out_factor2 * in_factor2)

            # SVD
            U, S, V = torch.svd(W)

            # Use top singular vectors for first factor
            s_sqrt = torch.sqrt(S[0])
            kronecker_layer.K1_factors[0].data.copy_(
                U[:, 0].reshape(out_factor1, in_factor1).to(kronecker_layer.K1_factors[0].device) * s_sqrt
            )
            kronecker_layer.K2_factors[0].data.copy_(
                V[:, 0].reshape(out_factor2, in_factor2).to(kronecker_layer.K2_factors[0].device) * s_sqrt
            )

            # For remaining factors, use additional singular vectors if available
            for i in range(1, num_factors):
                if i < min(S.size(0), 10):  # Use up to 10 singular vectors
                    s_sqrt = torch.sqrt(S[i])
                    kronecker_layer.K1_factors[i].data.copy_(
                        U[:, i].reshape(out_factor1, in_factor1).to(kronecker_layer.K1_factors[i].device) * s_sqrt
                    )
                    kronecker_layer.K2_factors[i].data.copy_(
                        V[:, i].reshape(out_factor2, in_factor2).to(kronecker_layer.K2_factors[i].device) * s_sqrt
                    )
        except Exception as e:
            logger.warning(f"SVD initialization failed: {e}. Using random initialization.")

        return kronecker_layer


def replace_linear_with_kronecker(model, max_candidate=32, num_factors=4, auto_factorize=False):
    """Replace linear layers in the model with Kronecker factorized layers"""
    layer_idx = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Create Kronecker layer from linear layer
            kronecker_layer = KroneckerLinear.from_linear(
                module, 
                num_factors=num_factors,
                layer_idx=layer_idx,
                auto_factorize=auto_factorize
            )

            # Ensure the Kronecker layer is on the same device as the original linear layer
            kronecker_layer = kronecker_layer.to(module.weight.device)
            # Find parent module to replace the linear layer
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, kronecker_layer)
                logger.info(f"Replaced linear layer with Kronecker layer: {name}")
            else:
                model.linear_layers[layer_idx] = kronecker_layer
                logger.info(f"Replaced linear layer with Kronecker layer: {name}")
            layer_idx += 1
            if layer_idx >= max_candidate:
                break

    return model


def evaluate_model(model, tokenizer, dataset, device, max_samples=100):
    """Evaluate model on dataset and return loss and perplexity"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    # Limit number of samples for evaluation
    eval_dataset = dataset.select(range(min(max_samples, len(dataset))))

    with torch.no_grad():
        for i in range(len(eval_dataset)):
            # Tokenize and prepare input
            inputs = tokenizer(eval_dataset[i]["text"], return_tensors="pt").to(device)

            # Skip samples that are too long
            if inputs["input_ids"].shape[1] > 1024:
                continue

            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])

            # Accumulate loss
            total_loss += outputs.loss.item() * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()

    # Calculate metrics
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return {"loss": avg_loss, "perplexity": perplexity}


def calibrate_kronecker(model_name="gpt2", dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", 
                        split="test", max_candidate=32, num_factors=4, auto_factorize=False):
    """Calibrate a model using Kronecker factorization

    Args:
        model_name: Name of the model to calibrate
        dataset_name: Name of the dataset to use for calibration
        dataset_config: Configuration of the dataset
        split: Split of the dataset to use
        max_candidate: Maximum number of linear layers to replace
        num_factors: Number of Kronecker factors to use
        auto_factorize: Whether to automatically determine factorization dimensions

    Returns:
        dict: Evaluation results including loss and perplexity
    """
    logger.info(f"Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    logger.info(f"Loading dataset {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)

    # Evaluate model before Kronecker factorization
    logger.info("Evaluating model before Kronecker factorization")
    before_eval = evaluate_model(model, tokenizer, dataset, device)
    logger.info(f"Before Kronecker: {before_eval}")

    # Replace linear layers with Kronecker factorized layers
    logger.info(f"Replacing linear layers with Kronecker factorized layers (max_candidate={max_candidate}, num_factors={num_factors})")
    model = replace_linear_with_kronecker(model, max_candidate=max_candidate, num_factors=num_factors)

    # Evaluate model after Kronecker factorization
    logger.info("Evaluating model after Kronecker factorization")
    after_eval = evaluate_model(model, tokenizer, dataset, device)
    logger.info(f"After Kronecker: {after_eval}")

    return after_eval
# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate a model using Kronecker factorization")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Name of the model to calibrate")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Name of the dataset to use for calibration")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Configuration of the dataset")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset to use")
    parser.add_argument("--max_candidate", type=int, default=32, help="Maximum number of linear layers to replace")
    parser.add_argument("--num_factors", type=int, default=4, help="Number of Kronecker factors to use")
    parser.add_argument("--auto_factorize", type=bool, default=False, help="Whether to automatically determine factorization dimensions")

    args = parser.parse_args()

    eval_result = calibrate_kronecker(
        model_name=args.model_name, 
        dataset_name=args.dataset_name, 
        dataset_config=args.dataset_config, 
        split=args.split, 
        max_candidate=args.max_candidate, 
        num_factors=args.num_factors,
        auto_factorize=args.auto_factorize
    )
    logger.info(f"Eval result: {eval_result}")
