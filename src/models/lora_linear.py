import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class LoRAStats:
    """Statistics about LoRA parameter reduction"""
    original_params: int
    lora_params: int
    params_saved: int
    reduction_ratio: float
    reduction_percentage: float
    lora_rank: int
    input_dim: int
    output_dim: int

    def __str__(self) -> str:
        return (
            f"LoRA Statistics:\n"
            f"Input dimension: {self.input_dim}\n"
            f"Output dimension: {self.output_dim}\n"
            f"LoRA rank: {self.lora_rank}\n"
            f"\nParameter counts:\n"
            f"Original parameters: {self.original_params:,}\n"
            f"LoRA parameters: {self.lora_params:,}\n"
            f"Parameters saved: {self.params_saved:,}\n"
            f"\nReduction metrics:\n"
            f"Reduction ratio: {self.reduction_ratio:.2f}x\n"
            f"Parameter reduction: {self.reduction_percentage:.2f}%"
        )


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer with a LoRA low-rank update.

    The forward pass computes:
        y = F.linear(x, W, b) + scaling * [F.linear(x, lora_A) @ lora_B.T]

    where:
      - W (and b) come from the original (frozen) linear layer.
      - lora_A (of shape: [r, in_features]) and lora_B (of shape: [out_features, r])
        are trainable parameters.
      - scaling = alpha / r.

    Only the LoRA parameters are updated during fine-tuning.

    Args:
        original_linear: The linear layer to wrap with LoRA.
        r: The rank of the low-rank approximation.
        alpha: Scaling factor for the LoRA update.
    """

    def __init__(self, original_linear: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        self.original_linear = original_linear
        # Freeze original parameters.
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Initialize lora_A with shape (r, in_features) and lora_B with shape (out_features, r).
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x):
        original_out = self.original_linear(x)
        # Compute LoRA update: F.linear(x, lora_A) yields shape (batch, r)
        lora_out = F.linear(x, self.lora_A)
        lora_update = lora_out @ self.lora_B.t()  # shape: (batch, out_features)
        return original_out + self.scaling * lora_update

    def get_stats(self) -> LoRAStats:
        """
        Compute parameter reduction statistics for this LoRA layer.

        Returns:
            LoRAStats: Statistics about parameter counts and reduction
        """
        in_features = self.original_linear.in_features
        out_features = self.original_linear.out_features

        # Original parameter count (weight matrix + bias)
        original_params = in_features * out_features
        if self.original_linear.bias is not None:
            original_params += out_features

        # LoRA parameter count (lora_A + lora_B)
        lora_params = (in_features * self.r) + (out_features * self.r)

        # Calculate reduction
        params_saved = original_params - lora_params
        reduction_ratio = original_params / lora_params
        reduction_percentage = (1 - lora_params / original_params) * 100

        return LoRAStats(
            original_params=original_params,
            lora_params=lora_params,
            params_saved=params_saved,
            reduction_ratio=reduction_ratio,
            reduction_percentage=reduction_percentage,
            lora_rank=self.r,
            input_dim=in_features,
            output_dim=out_features
        )

    def __repr__(self):
        return f"LoRALinear(in_features={self.original_linear.in_features}, " \
               f"out_features={self.original_linear.out_features}, " \
               f"rank={self.r}, alpha={self.alpha})\n" + \
            str(self.get_stats())


# Testing code
if __name__ == "__main__":
    # Example usage:
    layer = nn.Linear(768, 768)
    lora_layer = LoRALinear(layer, r=8)
    stats = lora_layer.get_stats()
    print(stats)  # This will print nicely formatted statistics
