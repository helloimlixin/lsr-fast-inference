import torch
import torch.nn as nn
from src.models.kronecker_linear import KroneckerLinear
from src.models.lora_linear import LoRALinear

def refine_kron_als_torch(W, A, B, num_iter=10):
    """
    Refines a single Kronecker factorization of W ≈ kron(A, B) using ALS.
    Args:
        W: Original weight matrix of shape (p*r, q*s).
        A: Factor matrix of shape (p, q).
        B: Factor matrix of shape (r, s).
        num_iter: Number of ALS iterations.
    Returns:
        Refined A and B.
    """
    p, q = A.shape
    r, s = B.shape
    # Reshape W to 4D: (p, r, q, s)
    W_4d = W.view(p, r, q, s)
    for _ in range(num_iter):
        B_norm_sq = torch.sum(B * B) + 1e-8
        A_new = torch.zeros_like(A)
        for i in range(p):
            for j in range(q):
                A_new[i, j] = torch.sum(W_4d[i, :, j, :] * B) / B_norm_sq
        A = A_new
        numerator = torch.zeros_like(B)
        denom = 0.0
        for i in range(p):
            for j in range(q):
                numerator += A[i, j] * W_4d[i, :, j, :]
                denom += A[i, j] ** 2
        B = numerator / (denom + 1e-8)
    return A, B

def factorize_linear_layer(layer, p, q, r, s, als_iter=10):
    """
    Factorizes an nn.Linear layer's weight using a single Kronecker product.
    First, it uses a Kron-SVD to initialize factors A and B, then optionally refines them with ALS.

    Args:
        layer: nn.Linear layer to factorize.
        p, q, r, s: Factorization dimensions (in_features must equal q*s and out_features equal p*r).
        als_iter: Number of ALS refinement iterations (set to 0 to skip refinement).

    Returns:
        A new KroneckerLinear layer approximating the original layer.
    """
    original_weight = layer.weight.data
    in_features = original_weight.shape[1]
    out_features = original_weight.shape[0]

    kron_layer = KroneckerLinear(in_features, out_features, p, q, r, s, bias=(layer.bias is not None))
    if layer.bias is not None:
        kron_layer.bias.data.copy_(layer.bias.data)

    # Reshape original weight:
    # Reshape to (p, r, q, s) then permute to (p, q, r, s) and reshape to (p*q, r*s)
    W_reshaped = original_weight.reshape(p, r, q, s).permute(0, 2, 1, 3).reshape(p * q, r * s)
    U, S, Vh = torch.linalg.svd(W_reshaped, full_matrices=False)

    # Use the top singular component for initialization.
    A = (U[:, 0] * torch.sqrt(S[0])).reshape(p, q)
    B = (Vh[0, :] * torch.sqrt(S[0])).reshape(r, s)

    # Optional ALS refinement
    if als_iter > 0:
        A, B = refine_kron_als_torch(original_weight, A, B, num_iter=als_iter)

    kron_layer.A.data.copy_(A)
    kron_layer.B.data.copy_(B)

    return kron_layer


def evaluate_factorization(original_layer, kron_layer, test_input=None):
    """
    Evaluate the quality of the factorization.

    Args:
        original_layer: Original nn.Linear layer
        kron_layer: Factorized KroneckerLinear layer
        test_input: Optional test input to check functional equivalence

    Returns:
        Dictionary with evaluation metrics
    """
    original_weight = original_layer.weight.data
    approx_weight = kron_layer.get_equivalent_weight()

    # Weight approximation error
    rel_error = torch.norm(original_weight - approx_weight) / torch.norm(original_weight)

    # Parameter counts
    original_params = np.prod(original_weight.shape)
    if original_layer.bias is not None:
        original_params += original_weight.shape[0]

    factorized_params = kron_layer.A.numel() + kron_layer.B.numel()
    if kron_layer.bias is not None:
        factorized_params += kron_layer.bias.numel()

    # Compression ratio
    compression_ratio = original_params / factorized_params

    # Functional comparison if test input is provided
    if test_input is not None:
        with torch.no_grad():
            original_output = original_layer(test_input)
            kron_output = kron_layer(test_input)
            functional_error = torch.norm(original_output - kron_output) / torch.norm(original_output)
    else:
        functional_error = None

    return {
        'relative_error': rel_error.item(),
        'original_params': original_params,
        'factorized_params': factorized_params,
        'compression_ratio': compression_ratio,
        'functional_error': functional_error
    }


def try_replace_linear_with_kronecker(layer, max_candidate, als_iter):
    """
    If the layer is a Linear layer and its dimensions can be factorized,
    replace it with a Kronecker-approximated version.
    The max_candidate parameter restricts the maximum allowed factor size.
    """
    if not isinstance(layer, nn.Linear):
        return layer

    out_features, in_features = layer.weight.shape
    # Find factor pairs for out_features and in_features
    def get_factors(n, max_candidate):
        pairs = []
        for a in range(1, max_candidate + 1):
            if n % a == 0:
                b = n // a
                pairs.append((a, b))
        return pairs

    out_factors = get_factors(out_features, max_candidate)
    in_factors = get_factors(in_features, max_candidate)
    if len(out_factors) == 0 or len(in_factors) == 0:
        return layer  # Cannot factorize
    # For simplicity, pick the last pair (largest factor within max_candidate)
    p, r = out_factors[-1]
    q, s = in_factors[-1]
    if p * r != out_features or q * s != in_features:
        return layer
    return factorize_linear_layer(layer, p, q, r, s, als_iter)


def replace_all_linear_with_kronecker(model, max_candidate, als_iter):
    for name, child in model.named_children():
        new_child = try_replace_linear_with_kronecker(child, max_candidate, als_iter)
        if isinstance(child, nn.Linear):
            new_child = try_replace_linear_with_kronecker(child, max_candidate, als_iter)
        setattr(model, name, new_child)
    return model


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