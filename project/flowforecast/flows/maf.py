"""Conditional Masked Autoregressive Flow layer (paper §2.1, p. 3).

MAF generalizes Real NVP: the transformation layer is an autoregressive
network. The Jacobian is triangular, so the determinant stays tractable.

The asymmetry the paper points out, and the reason LSTM-MAF beats
LSTM-Real-NVP on density but is slower to sample:
    forward (density evaluation) -> one parallel pass
    inverse (sampling)           -> sequential, D passes
"The inverse however, needed for generating samples, is sequential."

The paper defers the exact parameterization to Papamakarios et al. (2017).
Standard form:
    raw_mu^i, raw_alpha^i = MADE(x^{1:i-1}, h)
    alpha^i = B * tanh(raw_alpha^i / B)
    forward:  y^i = (x^i - mu^i) * exp(-alpha^i),  log|det J| = -sum_i alpha^i
    inverse:  x^i = y^i * exp(alpha^i) + mu^i      (dim by dim, in order)

The official pytorch-ts MAF leaves alpha unbounded. ``log_scale_bound=0`` is
therefore the paper-parity setting; a positive value is only a diagnostic
ablation and is not used by the reproduction configs.

The sequential inverse rebuilds x with torch.cat instead of writing into a
tensor in place: in-place index assignment on a tensor that is also fed back
into the MADE network breaks autograd (and silently produces wrong gradients in
older torch versions).
"""

import os

import torch
import torch.nn as nn

from .made import MADE


_DIAGNOSTICS_ENV = "FLOWFORECAST_MAF_DIAGNOSTICS"


def _diagnostics_enabled() -> bool:
    return os.getenv(_DIAGNOSTICS_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _tensor_diagnostic(name: str, value: torch.Tensor) -> str:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    if finite_values.numel():
        finite_min = float(finite_values.min().item())
        finite_max = float(finite_values.max().item())
    else:
        finite_min = float("nan")
        finite_max = float("nan")
    return (
        f"{name}: shape={tuple(detached.shape)}, dtype={detached.dtype}, "
        f"finite={int(finite.sum().item())}/{detached.numel()}, "
        f"nan={int(torch.isnan(detached).sum().item())}, "
        f"+inf={int(torch.isposinf(detached).sum().item())}, "
        f"-inf={int(torch.isneginf(detached).sum().item())}, "
        f"finite_range=[{finite_min:.6g}, {finite_max:.6g}]"
    )


def _require_finite(stage: str, **values: torch.Tensor) -> None:
    if all(bool(torch.isfinite(value).all().item()) for value in values.values()):
        return
    details = "; ".join(
        _tensor_diagnostic(name, value) for name, value in values.items()
    )
    raise FloatingPointError(f"MAF numerical diagnostic - {stage} - {details}")


def _abs_max(value: torch.Tensor) -> float:
    detached = value.detach()
    if detached.numel() == 0:
        return 0.0
    return float(detached.abs().max().item())


class ConditionalMAFLayer(nn.Module):
    conditional = True

    def __init__(self, features: int, cond_size: int, hidden: int = 100,
                 n_hidden: int = 2,
                 input_degrees: torch.Tensor | None = None,
                 log_scale_bound: float = 0.0,
                 mu_bound: float = 0.0):
        super().__init__()
        try:
            self.features = features
            self.cond_size = cond_size
            self.log_scale_bound = float(log_scale_bound)
            self.mu_bound = float(mu_bound)
            if self.log_scale_bound < 0.0 or self.mu_bound < 0.0:
                raise ValueError("log_scale_bound and mu_bound must be non-negative")
            self.made = MADE(features, cond_size, hidden, n_hidden,
                             input_degrees=input_degrees)
            self.input_degrees = self.made.input_degrees
            self._last_inverse_diagnostics: dict[str, float] = {}
        except Exception as e:
            raise Exception(f"ConditionalMAFLayer - __init__ - Init error: {e}") from e

    def _bound_log_scale(self, raw_alpha: torch.Tensor) -> torch.Tensor:
        if self.log_scale_bound == 0.0:
            return raw_alpha
        bound = self.log_scale_bound
        return bound * torch.tanh(raw_alpha / bound)

    def _bound_mu(self, raw_mu: torch.Tensor) -> torch.Tensor:
        if self.mu_bound == 0.0:
            return raw_mu
        return self.mu_bound * torch.tanh(raw_mu / self.mu_bound)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        try:
            diagnostics = _diagnostics_enabled()
            if diagnostics:
                inputs = {"x": x}
                if h is not None:
                    inputs["condition"] = h
                _require_finite("forward input", **inputs)
            raw_mu, raw_alpha = self.made(x, h)
            if diagnostics:
                _require_finite(
                    "forward MADE output", raw_mu=raw_mu,
                    raw_alpha=raw_alpha,
                )
            mu = self._bound_mu(raw_mu)
            alpha = self._bound_log_scale(raw_alpha)
            inverse_scale = torch.exp(-alpha)
            if diagnostics:
                _require_finite(
                    "forward exp(-alpha)", raw_alpha=raw_alpha, alpha=alpha,
                    inverse_scale=inverse_scale,
                )
            y = (x - mu) * inverse_scale
            log_det = -alpha.sum(dim=-1)
            if diagnostics:
                _require_finite("forward result", y=y, log_det=log_det)
            return y, log_det
        except Exception as e:
            raise Exception(f"ConditionalMAFLayer - forward - Execution error: {e}") from e

    def inverse(self, y: torch.Tensor, h: torch.Tensor | None = None):
        try:
            diagnostics = _diagnostics_enabled()
            if diagnostics:
                inputs = {"y": y}
                if h is not None:
                    inputs["condition"] = h
                _require_finite("inverse input", **inputs)
                inverse_diagnostics = {
                    "input_abs_max": _abs_max(y),
                    "raw_alpha_abs_max": 0.0,
                    "alpha_abs_max": 0.0,
                    "scale_max": 0.0,
                    "raw_mu_abs_max": 0.0,
                    "mu_abs_max": 0.0,
                    "output_abs_max": 0.0,
                }
            x = torch.zeros_like(y)
            for degree in self.input_degrees:
                i = int(degree)
                raw_mu, raw_alpha = self.made(x, h)
                mu = self._bound_mu(raw_mu)
                mu_i = mu[..., i]
                raw_mu_i = raw_mu[..., i]
                raw_alpha_i = raw_alpha[..., i]
                alpha = self._bound_log_scale(raw_alpha)
                alpha_i = alpha[..., i]
                if diagnostics:
                    _require_finite(
                        f"inverse MADE output at dimension {i}",
                        partial_x=x, raw_mu_i=raw_mu_i, mu_i=mu_i,
                        raw_alpha_i=raw_alpha_i, alpha_i=alpha_i,
                    )
                scale_i = torch.exp(alpha_i)
                if diagnostics:
                    inverse_diagnostics["raw_alpha_abs_max"] = max(
                        inverse_diagnostics["raw_alpha_abs_max"],
                        _abs_max(raw_alpha_i),
                    )
                    inverse_diagnostics["alpha_abs_max"] = max(
                        inverse_diagnostics["alpha_abs_max"],
                        _abs_max(alpha_i),
                    )
                    inverse_diagnostics["scale_max"] = max(
                        inverse_diagnostics["scale_max"],
                        float(scale_i.detach().max().item()),
                    )
                    inverse_diagnostics["mu_abs_max"] = max(
                        inverse_diagnostics["mu_abs_max"],
                        _abs_max(mu_i),
                    )
                    inverse_diagnostics["raw_mu_abs_max"] = max(
                        inverse_diagnostics["raw_mu_abs_max"],
                        _abs_max(raw_mu_i),
                    )
                    _require_finite(
                        f"inverse exp(alpha) at dimension {i}",
                        raw_alpha_i=raw_alpha_i, alpha_i=alpha_i,
                        scale_i=scale_i,
                    )
                x_i = y[..., i] * scale_i + mu_i
                if diagnostics:
                    _require_finite(
                        f"inverse reconstructed value at dimension {i}",
                        y_i=y[..., i], scale_i=scale_i, mu_i=mu_i, x_i=x_i,
                    )
                x = torch.cat([x[..., :i], x_i.unsqueeze(-1), x[..., i + 1:]],
                              dim=-1)
            if diagnostics:
                inverse_diagnostics["output_abs_max"] = _abs_max(x)
                self._last_inverse_diagnostics = inverse_diagnostics
            return x
        except Exception as e:
            raise Exception(f"ConditionalMAFLayer - inverse - Execution error: {e}") from e

    @property
    def last_inverse_diagnostics(self) -> dict[str, float]:
        return dict(self._last_inverse_diagnostics)
