from collections.abc import Callable

import torch.nn.functional as F
from torch import Tensor

__all__ = ["ACTIVATIONS"]

ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "relu": F.relu,
    "rrelu": F.rrelu,
    "hardtanh": F.hardtanh,
    "relu6": F.relu6,
    "sigmoid": F.sigmoid,
    "hardsigmoid": F.hardsigmoid,
    "tanh": F.tanh,
    "silu": F.silu,
    "mish": F.mish,
    "hardswish": F.hardswish,
    "elu": F.elu,
    "celu": F.celu,
    "selu": F.selu,
    "glu": F.glu,
    "gelu": F.gelu,
    "hardshrink": F.hardshrink,
    "leakyrelu": F.leaky_relu,
    "softplus": F.softplus,
    "softshrink": F.softshrink,
    "softsign": F.softsign,
    "tanhshrink": F.tanhshrink,
}
