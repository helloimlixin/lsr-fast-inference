import torch
import torch.nn as nn

from src.models.kronecker_linear import KroneckerLinear
from src.models.lora_linear import LoRALinear


EPSILON = 1e-8
DEFAULT_FACTORIZATION_OBJECTIVE = "balanced"
DEFAULT_TILE_MULTIPLE = 16
DEFAULT_MIN_FACTOR_SIZE = 32
DEFAULT_KRONECKER_IMPLEMENTATION = "gemm"


def refine_kron_als_torch(weight, factor_a, factor_b, num_iter=10):
    """Refine a Kronecker factorization with alternating least squares."""
    p, q = factor_a.shape
    r, s = factor_b.shape
    weight_4d = weight.reshape(p, r, q, s)

    for _ in range(max(num_iter, 0)):
        factor_b_norm = factor_b.pow(2).sum().clamp_min(EPSILON)
        factor_a = torch.einsum("irqs,rs->iq", weight_4d, factor_b) / factor_b_norm

        factor_a_norm = factor_a.pow(2).sum().clamp_min(EPSILON)
        factor_b = torch.einsum("iq,irqs->rs", factor_a, weight_4d) / factor_a_norm

    return factor_a, factor_b


def factorize_linear_layer(
    layer,
    p,
    q,
    r,
    s,
    als_iter=10,
    implementation=DEFAULT_KRONECKER_IMPLEMENTATION,
):
    """Approximate a linear layer with a single Kronecker product."""
    original_weight = layer.weight.detach()
    out_features, in_features = original_weight.shape
    factorization_dtype = torch.float32

    kron_layer = KroneckerLinear(
        in_features=in_features,
        out_features=out_features,
        p=p,
        q=q,
        r=r,
        s=s,
        bias=layer.bias is not None,
        implementation=implementation,
    ).to(device=original_weight.device, dtype=original_weight.dtype)

    if layer.bias is not None:
        kron_layer.bias.data.copy_(layer.bias.detach())

    # Run SVD/ALS in float32 even when the model is loaded in bf16/fp16.
    # This avoids unstable or unsupported low-precision linear algebra paths.
    factorization_weight = original_weight.to(dtype=factorization_dtype)
    reshaped_weight = (
        factorization_weight.reshape(p, r, q, s)
        .permute(0, 2, 1, 3)
        .reshape(p * q, r * s)
    )
    left, singular_values, right = torch.linalg.svd(reshaped_weight, full_matrices=False)

    scale = torch.sqrt(singular_values[0])
    factor_a = (left[:, 0] * scale).reshape(p, q)
    factor_b = (right[0, :] * scale).reshape(r, s)

    if als_iter > 0:
        factor_a, factor_b = refine_kron_als_torch(
            factorization_weight,
            factor_a,
            factor_b,
            num_iter=als_iter,
        )

    kron_layer.A.data.copy_(factor_a.to(dtype=original_weight.dtype))
    kron_layer.B.data.copy_(factor_b.to(dtype=original_weight.dtype))
    return kron_layer


def evaluate_factorization(original_layer, kron_layer, test_input=None):
    """Report approximation quality for a Kronecker-factorized layer."""
    original_weight = original_layer.weight.detach()
    approx_weight = kron_layer.get_equivalent_weight()

    rel_error = torch.norm(original_weight - approx_weight) / torch.norm(original_weight).clamp_min(EPSILON)

    original_params = original_weight.numel()
    if original_layer.bias is not None:
        original_params += original_layer.bias.numel()

    factorized_params = kron_layer.A.numel() + kron_layer.B.numel()
    if kron_layer.bias is not None:
        factorized_params += kron_layer.bias.numel()

    functional_error = None
    if test_input is not None:
        with torch.no_grad():
            original_output = original_layer(test_input)
            kron_output = kron_layer(test_input)
        functional_error = (
            torch.norm(original_output - kron_output)
            / torch.norm(original_output).clamp_min(EPSILON)
        ).item()

    return {
        "relative_error": rel_error.item(),
        "original_params": original_params,
        "factorized_params": factorized_params,
        "compression_ratio": original_params / factorized_params,
        "functional_error": functional_error,
    }


def _get_factor_pairs(value, max_candidate):
    limit = min(value, max_candidate)
    return [
        (left, value // left)
        for left in range(1, limit + 1)
        if value % left == 0
    ]


def _alignment_penalty(value, tile_multiple):
    if tile_multiple <= 1:
        return 0

    remainder = value % tile_multiple
    return min(remainder, tile_multiple - remainder)


def _score_factorization_dims(
    dims,
    original_params,
    factorization_objective,
    tile_multiple,
    min_factor_size,
):
    p, q, r, s = dims
    factorized_params = (p * q) + (r * s)
    imbalance = abs(p - r) + abs(q - s)
    min_dim = min(dims)
    alignment_penalty = sum(
        _alignment_penalty(value, tile_multiple)
        for value in dims
    )
    size_penalty = sum(
        max(0, min_factor_size - value)
        for value in dims
    )
    compression_penalty = float(factorized_params) / float(original_params)

    if factorization_objective == "compression":
        return (factorized_params, imbalance, -min_dim)

    if factorization_objective == "latency":
        return (alignment_penalty, size_penalty, compression_penalty, imbalance)

    if factorization_objective == "balanced":
        return (alignment_penalty, compression_penalty, size_penalty, imbalance)

    raise ValueError(
        "Unsupported factorization objective: {}".format(factorization_objective)
    )


def _select_factorization_dims(
    layer,
    max_candidate,
    factorization_objective=DEFAULT_FACTORIZATION_OBJECTIVE,
    tile_multiple=DEFAULT_TILE_MULTIPLE,
    min_factor_size=DEFAULT_MIN_FACTOR_SIZE,
):
    out_features, in_features = layer.weight.shape
    original_params = out_features * in_features
    best_candidate = None

    for p, r in _get_factor_pairs(out_features, max_candidate):
        for q, s in _get_factor_pairs(in_features, max_candidate):
            factorized_params = (p * q) + (r * s)
            if factorized_params >= original_params:
                continue

            candidate = (
                _score_factorization_dims(
                    (p, q, r, s),
                    original_params,
                    factorization_objective,
                    tile_multiple,
                    min_factor_size,
                ),
                (p, q, r, s),
            )
            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate

    if best_candidate is None:
        return None

    return best_candidate[-1]


def try_replace_linear_with_kronecker(
    layer,
    max_candidate,
    als_iter,
    factorization_objective=DEFAULT_FACTORIZATION_OBJECTIVE,
    tile_multiple=DEFAULT_TILE_MULTIPLE,
    min_factor_size=DEFAULT_MIN_FACTOR_SIZE,
    implementation=DEFAULT_KRONECKER_IMPLEMENTATION,
):
    """Return a Kronecker approximation when a useful factorization exists."""
    if not isinstance(layer, nn.Linear):
        return layer

    dims = _select_factorization_dims(
        layer,
        max_candidate,
        factorization_objective=factorization_objective,
        tile_multiple=tile_multiple,
        min_factor_size=min_factor_size,
    )
    if dims is None:
        return layer

    return factorize_linear_layer(
        layer,
        *dims,
        als_iter=als_iter,
        implementation=implementation,
    )


def replace_classifier_with_kronecker(
    layer,
    max_candidate,
    als_iter,
    factorization_objective=DEFAULT_FACTORIZATION_OBJECTIVE,
    tile_multiple=DEFAULT_TILE_MULTIPLE,
    min_factor_size=DEFAULT_MIN_FACTOR_SIZE,
    implementation=DEFAULT_KRONECKER_IMPLEMENTATION,
):
    """Replace a classifier head with a Kronecker approximation when possible."""
    return try_replace_linear_with_kronecker(
        layer,
        max_candidate,
        als_iter,
        factorization_objective=factorization_objective,
        tile_multiple=tile_multiple,
        min_factor_size=min_factor_size,
        implementation=implementation,
    )


def _matches_target_modules(full_name, target_modules):
    if not target_modules:
        return True
    return any(token in full_name for token in target_modules)


def replace_all_linear_with_kronecker(
    module,
    max_candidate,
    als_iter,
    target_modules=None,
    factorization_objective=DEFAULT_FACTORIZATION_OBJECTIVE,
    tile_multiple=DEFAULT_TILE_MULTIPLE,
    min_factor_size=DEFAULT_MIN_FACTOR_SIZE,
    implementation=DEFAULT_KRONECKER_IMPLEMENTATION,
    prefix="",
):
    """Recursively replace eligible linear layers with Kronecker layers."""
    if isinstance(module, nn.Linear):
        if not _matches_target_modules(prefix, target_modules):
            return module
        return try_replace_linear_with_kronecker(
            module,
            max_candidate,
            als_iter,
            factorization_objective=factorization_objective,
            tile_multiple=tile_multiple,
            min_factor_size=min_factor_size,
            implementation=implementation,
        )

    for name, child in module.named_children():
        child_prefix = "{}.{}".format(prefix, name) if prefix else name
        setattr(
            module,
            name,
            replace_all_linear_with_kronecker(
                child,
                max_candidate,
                als_iter,
                target_modules=target_modules,
                factorization_objective=factorization_objective,
                tile_multiple=tile_multiple,
                min_factor_size=min_factor_size,
                implementation=implementation,
                prefix=child_prefix,
            ),
        )
    return module


def replace_linear_with_lora(module, r, alpha):
    """Recursively wrap linear layers with LoRA adapters."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r, alpha))
            continue
        replace_linear_with_lora(child, r, alpha)
    return module


def get_kronecker_stats(module):
    """Summarize parameter counts for Kronecker layers in a module tree."""
    stats = {
        "kronecker_layers": 0,
        "kronecker_original_params": 0,
        "kronecker_factorized_params": 0,
    }

    for child in module.modules():
        if not isinstance(child, KroneckerLinear):
            continue

        original_params = child.in_features * child.out_features
        factorized_params = child.A.numel() + child.B.numel()

        if child.bias is not None:
            original_params += child.bias.numel()
            factorized_params += child.bias.numel()

        stats["kronecker_layers"] += 1
        stats["kronecker_original_params"] += original_params
        stats["kronecker_factorized_params"] += factorized_params

    if stats["kronecker_layers"]:
        saved_params = (
            stats["kronecker_original_params"] - stats["kronecker_factorized_params"]
        )
        stats["kronecker_params_saved"] = saved_params
        stats["kronecker_reduction_pct"] = round(
            100.0 * saved_params / stats["kronecker_original_params"],
            2,
        )

    return stats
