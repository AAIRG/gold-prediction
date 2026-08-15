"""ConditionalFlow: stacks K conditional blocks (paper §2.1, eq. 1).

Models p_X(x | h) via the change of variables formula:

    log p_X(x) = log p_Z(z) + sum_{i=1}^{K} log|det(dy_i / dy_{i-1})|    (eq. 1)

Block order follows Figure 1 of the paper:
    x -> Coupling 1 -> BatchNorm 1 -> ... -> Coupling K -> BatchNorm K -> z

K = 5 per Appendix D.2. Base distribution p_Z is an isotropic Gaussian
(§2.1: "some simple distribution p_Z (e.g. an isotropic Gaussian)").

Two structural details that decide whether the stack is actually expressive:

  * Real NVP: the binary mask alternates every block (mask, 1 - mask), so every
    dimension is transformed in half of the blocks. A dimension that is never
    transformed keeps a correct-looking log_det while its samples stay pure
    noise (roadmap section 7).
  * MAF: an autoregressive layer has ONE fixed variable order, so stacking K of
    them without shuffling leaves the first dimension unconditioned forever.
    A `Permutation` bijection (order reversal, log_det = 0) is therefore
    inserted between MAF blocks -- this is what Papamakarios et al. (2017) do,
    and the paper's MAF variant inherits it.

All blocks share one interface, so log_prob / sample are single loops:
    forward(x, h) -> (y, log_det)
    inverse(y, h) -> x
Blocks that ignore the conditioning (BatchNorm, Permutation) simply accept and
drop `h`.
"""

import os

import torch
import torch.nn as nn

from .batchnorm import BatchNormFlow
from .coupling import ConditionalCouplingLayer
from .maf import ConditionalMAFLayer


def _diagnostics_enabled() -> bool:
    return os.getenv("FLOWFORECAST_MAF_DIAGNOSTICS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _abs_max(value: torch.Tensor) -> float:
    detached = value.detach()
    if detached.numel() == 0:
        return 0.0
    return float(detached.abs().max().item())


def _print_inverse_diagnostic(reverse_index: int, block: nn.Module,
                              value: torch.Tensor) -> None:
    details = [f"output_abs_max={_abs_max(value):.6g}"]
    if isinstance(block, ConditionalMAFLayer):
        details.extend(
            f"{name}={measurement:.6g}"
            for name, measurement in block.last_inverse_diagnostics.items()
            if name != "output_abs_max"
        )
    elif isinstance(block, BatchNormFlow):
        details.extend([
            f"running_var_min={float(block.running_var.detach().min().item()):.6g}",
            f"running_var_max={float(block.running_var.detach().max().item()):.6g}",
            f"log_gamma_abs_max={_abs_max(block.log_gamma):.6g}",
        ])
    print(
        "[maf-diagnostic] inverse block "
        f"{reverse_index} {type(block).__name__} | " + ", ".join(details),
        flush=True,
    )


class Permutation(nn.Module):
    """Fixed permutation of the D dimensions. Volume preserving: log_det = 0."""

    conditional = False

    def __init__(self, features: int, permutation: torch.Tensor | None = None):
        super().__init__()
        try:
            if permutation is None:
                permutation = torch.arange(features - 1, -1, -1)  # reverse order
            permutation = permutation.long()
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(features, dtype=torch.long)

            self.features = features
            self.register_buffer('permutation', permutation)
            self.register_buffer('inverse_permutation', inverse)
        except Exception as e:
            raise Exception(f"Permutation - __init__ - Init error: {e}") from e

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        try:
            y = x[..., self.permutation]
            log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            return y, log_det
        except Exception as e:
            raise Exception(f"Permutation - forward - Execution error: {e}") from e

    def inverse(self, y: torch.Tensor, h: torch.Tensor | None = None):
        try:
            return y[..., self.inverse_permutation]
        except Exception as e:
            raise Exception(f"Permutation - inverse - Execution error: {e}") from e


class ConditionalFlow(nn.Module):
    def __init__(self, features: int, cond_size: int, num_blocks: int = 5,
                 hidden: int = 100, block_type: str = "realnvp",
                 batch_norm: bool = True, bn_momentum: float = 0.1,
                 bn_min_var: float = 1e-4,
                 maf_log_scale_bound: float = 3.0,
                 maf_mu_bound: float = 10.0):
        super().__init__()
        try:
            self.features = features
            self.cond_size = cond_size
            self.num_blocks = num_blocks
            self.block_type = str(block_type).lower()

            self.blocks = nn.ModuleList()

            maf_degrees = None
            for i in range(num_blocks):
                if self.block_type == "realnvp":
                    mask = (torch.arange(features) % 2).float()
                    if i % 2:
                        mask = 1.0 - mask
                    self.blocks.append(
                        ConditionalCouplingLayer(features, cond_size, hidden, mask)
                    )
                elif self.block_type == "maf":
                    maf = ConditionalMAFLayer(
                        features, cond_size, hidden,
                        input_degrees=maf_degrees,
                        log_scale_bound=maf_log_scale_bound,
                        mu_bound=maf_mu_bound,
                    )
                    self.blocks.append(maf)
                    maf_degrees = maf.input_degrees.flip(0)
                else:
                    raise ValueError(f"Unknown block type {block_type}")

                if batch_norm:
                    self.blocks.append(
                        BatchNormFlow(features, momentum=bn_momentum,
                                      min_var=bn_min_var)
                    )


            self.register_buffer('base_mean', torch.zeros(features))
            self.register_buffer('base_var', torch.ones(features))
        except Exception as e:
            raise Exception(f"ConditionalFlow - __init__ - Init error: {e}") from e

    @property
    def base_dist(self) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.base_mean, torch.sqrt(self.base_var))

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """x -> z with the accumulated log|det J| (eq. 1)."""
        try:
            log_det_sum = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            z = x
            for block in self.blocks:
                z, log_det = block(z, h)
                log_det_sum = log_det_sum + log_det
            return z, log_det_sum
        except Exception as e:
            raise Exception(f"ConditionalFlow - forward - Execution error: {e}") from e

    def log_prob(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        try:
            z, log_det_sum = self.forward(x, h)
            log_p_z = self.base_dist.log_prob(z).sum(dim=-1)
            return log_p_z + log_det_sum
        except Exception as e:
            raise Exception(f"ConditionalFlow - log_prob - Execution error: {e}") from e

    def sample(self, h: torch.Tensor | None = None, num_samples: int = 1) -> torch.Tensor:
        """One draw per row of `h` (§4.3: no temperature scaling of p_Z).

        h [B, cond_size] -> [B, D]. If h has a single row (or is None) and
        num_samples > 1, it is expanded to [num_samples, D].
        """
        try:
            if h is not None and h.dim() == 1:
                h = h.unsqueeze(0)

            if h is None:
                batch_size = num_samples
            else:
                batch_size = h.shape[0]
                if num_samples > 1 and batch_size == 1:
                    h = h.expand(num_samples, -1)
                    batch_size = num_samples

            device = self.base_mean.device if h is None else h.device
            z = self.base_dist.sample((batch_size,)).to(device)

            x = z
            diagnostics = _diagnostics_enabled()
            if diagnostics:
                print(
                    f"[maf-diagnostic] base sample | abs_max={_abs_max(x):.6g}",
                    flush=True,
                )
            for reverse_index, block in enumerate(reversed(self.blocks)):
                x = block.inverse(x, h)
                if diagnostics:
                    _print_inverse_diagnostic(reverse_index, block, x)
                if diagnostics and not bool(torch.isfinite(x).all().item()):
                    finite = torch.isfinite(x)
                    finite_values = x.detach()[finite]
                    finite_range = (
                        f"[{float(finite_values.min().item()):.6g}, "
                        f"{float(finite_values.max().item()):.6g}]"
                        if finite_values.numel() else "[nan, nan]"
                    )
                    raise FloatingPointError(
                        "ConditionalFlow numerical diagnostic - inverse block "
                        f"{reverse_index} ({type(block).__name__}) produced "
                        f"non-finite values: finite="
                        f"{int(finite.sum().item())}/{x.numel()}, "
                        f"nan={int(torch.isnan(x).sum().item())}, "
                        f"+inf={int(torch.isposinf(x).sum().item())}, "
                        f"-inf={int(torch.isneginf(x).sum().item())}, "
                        f"finite_range={finite_range}"
                    )
            return x
        except Exception as e:
            raise Exception(f"ConditionalFlow - sample - Execution error: {e}") from e
