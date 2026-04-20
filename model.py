from __future__ import annotations

from typing import Iterable, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """A linear layer with learnable element-wise gates over weights."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / fan_in**0.5
        nn.init.uniform_(self.bias, -bound, bound)
        # Start gates around 0.5 so the model can learn both keep/prune directions.
        nn.init.zeros_(self.gate_scores)

    def gates(self) -> Tensor:
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: Tensor) -> Tensor:
        gates = self.gates()
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


class PrunableMLP(nn.Module):
    """Simple MLP for CIFAR-10 using only PrunableLinear layers."""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int], num_classes: int = 10) -> None:
        super().__init__()
        hidden_list: List[int] = list(hidden_dims)
        dims = [input_dim] + hidden_list + [num_classes]

        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(PrunableLinear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def prunable_layers(self) -> List[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> Tensor:
        return sum(layer.gates().sum() for layer in self.prunable_layers())

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        all_gates = self.all_gates()
        if all_gates.numel() == 0:
            return 0.0
        return float((all_gates < threshold).float().mean().item() * 100.0)

    def all_gates(self) -> Tensor:
        return torch.cat([layer.gates().detach().flatten() for layer in self.prunable_layers()], dim=0)


class BaselineMLP(nn.Module):
    """Non-prunable baseline MLP with matching architecture."""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int], num_classes: int = 10) -> None:
        super().__init__()
        hidden_list: List[int] = list(hidden_dims)
        dims = [input_dim] + hidden_list + [num_classes]

        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
