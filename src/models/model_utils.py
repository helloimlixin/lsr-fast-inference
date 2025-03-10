import torch.nn as nn
from src.models.kronecker_linear import KroneckerLinear
from src.models.lora_linear import LoRALinear


def get_factor_pairs(n, max_candidate):
    """
    Get all pairs of integers (a, b) where a * b = n and a <= max_candidate.

    Args:
        n: The number to factorize.
        max_candidate: The maximum value for the first factor.

    Returns:
        List of tuples (a, b) where a * b = n and a <= max_candidate.
    """
    pairs = []
    for a in range(1, min(max_candidate + 1, n + 1)):
        if n % a == 0:
            b = n // a
            pairs.append((a, b))
    return pairs


def try_replace_linear_with_kronecker(layer, max_candidate, als_iter):
    """
    If layer is nn.Linear and its dimensions can be factorized into two factors,
    replace it with a KroneckerLinear.

    Args:
        layer: The layer to potentially replace.
        max_candidate: Maximum value for the first factor in factorization.
        als_iter: Number of alternating least squares iterations (not used in this implementation).

    Returns:
        Original layer or KroneckerLinear replacement.
    """
    if not isinstance(layer, nn.Linear):
        return layer

    out_features, in_features = layer.weight.shape
    out_factors = get_factor_pairs(out_features, max_candidate)
    in_factors = get_factor_pairs(in_features, max_candidate)

    if len(out_factors) == 0 or len(in_factors) == 0:
        return layer

    # Choose the last candidate.
    p, r = out_factors[-1]
    q, s = in_factors[-1]

    if p * r != out_features or q * s != in_features:
        return layer

    return KroneckerLinear(
        in_features,
        out_features,
        p, q, r, s,
        bias=(layer.bias is not None)
    )


def replace_classifier_with_kronecker(classifier, max_candidate=32, als_iter=10):
    """
    Recursively traverse the classifier module and replace nn.Linear layers with
    their Kronecker-approximated versions.

    Args:
        classifier: The classifier module to modify.
        max_candidate: Maximum value for the first factor in factorization.
        als_iter: Number of alternating least squares iterations.

    Returns:
        Modified classifier module.
    """
    for name, child in classifier.named_children():
        new_child = replace_classifier_with_kronecker(child, max_candidate, als_iter)
        if isinstance(child, nn.Linear):
            new_child = try_replace_linear_with_kronecker(child, max_candidate, als_iter)
        setattr(classifier, name, new_child)
    return classifier


def replace_linear_with_lora(module: nn.Module, r: int, alpha: float):
    """
    Recursively traverse module and replace each nn.Linear layer with a LoRALinear.
    (This is applied to linear layers that are not already replaced by the Kronecker method.)

    Args:
        module: The module to modify.
        r: Rank for LoRA approximation.
        alpha: Scaling factor for LoRA update.

    Returns:
        Modified module.
    """
    for name, child in module.named_children():
        replace_linear_with_lora(child, r, alpha)
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r, alpha))
    return module