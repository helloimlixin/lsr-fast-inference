import math
import torch
import torch.nn as nn

class KroneckerLinear(nn.Module):
    """
    Approximates an nn.Linear layer using a single Kronecker product factorization:
        W ≈ A ⊗ B.

    Args:
        in_features: Must equal q * s.
        out_features: Must equal p * r.
        p, q: Dimensions for factor matrix A.
        r, s: Dimensions for factor matrix B.
        bias: Whether to include a bias.
    """
    def __init__(self, in_features, out_features, p, q, r, s, bias=True):
        super().__init__()
        assert in_features == q * s, f"in_features must equal q*s (got {in_features} != {q*s})"
        assert out_features == p * r, f"out_features must equal p*r (got {out_features} != {p*r})"

        self.in_features = in_features
        self.out_features = out_features
        self.p, self.q, self.r, self.s = p, q, r, s

        self.A = nn.Parameter(torch.Tensor(p, q))
        self.B = nn.Parameter(torch.Tensor(r, s))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.q * self.s
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        batch_size = x_flat.size(0)
        # Reshape input to (batch, q, s)
        x_reshaped = x_flat.view(batch_size, self.q, self.s)
        # Expand factors to match batch dimension.
        A_exp = self.A.unsqueeze(0).expand(batch_size, self.p, self.q)
        B_exp = self.B.unsqueeze(0).expand(batch_size, self.r, self.s)
        # Compute: output = A @ (x_reshaped @ B^T)
        temp = torch.bmm(x_reshaped, B_exp.transpose(1, 2))  # shape: (batch, q, r)
        output = torch.bmm(A_exp, temp)  # shape: (batch, p, r)
        output = output.view(batch_size, self.p * self.r)
        if self.bias is not None:
            output = output + self.bias
        new_shape = orig_shape[:-1] + (self.p * self.r,)
        return output.view(new_shape)

    def get_equivalent_weight(self):
        """Return the full approximated weight matrix as A ⊗ B."""
        return torch.kron(self.A, self.B)