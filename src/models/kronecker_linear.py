import math
import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_IMPLEMENTATIONS = ("gemm", "einsum")


class KroneckerLinear(nn.Module):
    """Approximate a linear layer with a single Kronecker product."""

    def __init__(
        self,
        in_features,
        out_features,
        p,
        q,
        r,
        s,
        bias=True,
        implementation="gemm",
    ):
        super().__init__()
        if in_features != q * s:
            raise ValueError("in_features must equal q * s")
        if out_features != p * r:
            raise ValueError("out_features must equal p * r")
        if implementation not in VALID_IMPLEMENTATIONS:
            raise ValueError(
                "implementation must be one of {}".format(VALID_IMPLEMENTATIONS)
            )

        self.in_features = in_features
        self.out_features = out_features
        self.p, self.q, self.r, self.s = p, q, r, s
        self.implementation = implementation

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
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def _forward_einsum(self, x):
        output_shape = tuple(x.shape[:-1]) + (self.out_features,)
        x = x.reshape(-1, self.q, self.s)

        intermediate = torch.einsum("bqs,rs->bqr", x, self.B)
        return torch.einsum("pq,bqr->bpr", self.A, intermediate).reshape(output_shape)

    def _forward_gemm(self, x):
        output_shape = tuple(x.shape[:-1]) + (self.out_features,)
        x = x.reshape(-1, self.q, self.s)
        tokens = x.size(0)

        first_stage = F.linear(
            x.reshape(tokens * self.q, self.s),
            self.B,
        )
        second_stage_input = (
            first_stage.reshape(tokens, self.q, self.r)
            .transpose(1, 2)
            .contiguous()
            .reshape(tokens * self.r, self.q)
        )
        second_stage = F.linear(second_stage_input, self.A)
        return (
            second_stage.reshape(tokens, self.r, self.p)
            .transpose(1, 2)
            .contiguous()
            .reshape(output_shape)
        )

    def forward(self, x):
        if self.implementation == "gemm":
            output = self._forward_gemm(x)
        else:
            output = self._forward_einsum(x)

        if self.bias is not None:
            output = output + self.bias
        return output

    def get_equivalent_weight(self):
        """Return the full approximated weight matrix as A ⊗ B."""
        return torch.kron(self.A, self.B)

    def extra_repr(self):
        return (
            "in_features={}, out_features={}, factors=({}, {}) x ({}, {}), bias={}, implementation={}".format(
                self.in_features,
                self.out_features,
                self.p,
                self.q,
                self.r,
                self.s,
                self.bias is not None,
                self.implementation,
            )
        )
