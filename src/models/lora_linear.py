import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAStats(object):
    """Statistics about LoRA parameter reduction."""

    def __init__(
        self,
        original_params,
        lora_params,
        params_saved,
        reduction_ratio,
        reduction_percentage,
        lora_rank,
        input_dim,
        output_dim,
    ):
        self.original_params = original_params
        self.lora_params = lora_params
        self.params_saved = params_saved
        self.reduction_ratio = reduction_ratio
        self.reduction_percentage = reduction_percentage
        self.lora_rank = lora_rank
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __str__(self):
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
    """Wrap an existing linear layer with a trainable low-rank update."""

    def __init__(self, original_linear, r, alpha=1.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")

        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x):
        lora_update = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return self.original_linear(x) + (self.scaling * lora_update)

    def get_stats(self):
        """
        Compute parameter reduction statistics for this LoRA layer.
        """
        in_features = self.original_linear.in_features
        out_features = self.original_linear.out_features

        original_params = in_features * out_features
        if self.original_linear.bias is not None:
            original_params += out_features

        lora_params = (in_features * self.r) + (out_features * self.r)

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

    def extra_repr(self):
        return "in_features={}, out_features={}, rank={}, alpha={}".format(
            self.original_linear.in_features,
            self.original_linear.out_features,
            self.r,
            self.alpha,
        )
