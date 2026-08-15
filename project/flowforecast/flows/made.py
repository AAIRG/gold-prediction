"""Conditional MADE matching the official pytorch-ts MAF construction."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor,
                 cond_size: int = 0):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", mask)
        self.cond_weight = nn.Parameter(
            torch.rand(out_features, cond_size) / math.sqrt(cond_size)
        ) if cond_size > 0 else None

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None):
        output = F.linear(x, self.weight * self.mask, self.bias)
        if self.cond_weight is not None and condition is not None:
            output = output + F.linear(condition, self.cond_weight)
        return output


def _masks(features: int, hidden: int, n_hidden: int,
           input_degrees: torch.Tensor | None = None):
    degrees = [torch.arange(features) if input_degrees is None else input_degrees]
    for _ in range(n_hidden + 1):
        degrees.append(torch.arange(hidden) % max(1, features - 1))
    output_degrees = degrees[0] % features - 1
    degrees.append(output_degrees)
    masks = [
        (following.unsqueeze(-1) >= previous.unsqueeze(0)).float()
        for previous, following in zip(degrees[:-1], degrees[1:])
    ]
    return masks, degrees[0]


class MADE(nn.Module):
    def __init__(self, features: int, cond_size: int, hidden: int = 100,
                 n_hidden: int = 2,
                 input_degrees: torch.Tensor | None = None):
        super().__init__()
        try:
            self.features = features
            self.cond_size = cond_size
            masks, degrees = _masks(features, hidden, n_hidden, input_degrees)
            self.register_buffer("input_degrees", degrees)
            self.input_layer = MaskedLinear(
                features, hidden, masks[0], cond_size=cond_size
            )
            layers = []
            for mask in masks[1:-1]:
                layers.extend([nn.ReLU(), MaskedLinear(hidden, hidden, mask)])
            layers.append(nn.ReLU())
            output_layer = MaskedLinear(
                hidden, 2 * features, masks[-1].repeat(2, 1)
            )
            # Start each MAF block as an identity transform. This prevents the
            # first Transformer-conditioned update from exploding location/scale.
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
            layers.append(output_layer)
            self.layers = nn.ModuleList(layers)
        except Exception as e:
            raise Exception(f"MADE - __init__ - Init error: {e}") from e

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        try:
            output = self.input_layer(x, h)
            for layer in self.layers:
                output = layer(output, h) if isinstance(layer, MaskedLinear) \
                    else layer(output)
            return output.chunk(2, dim=-1)
        except Exception as e:
            raise Exception(f"MADE - forward - Execution error: {e}") from e
