import torch
import torch.nn as nn
import torch.nn.functional as F


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