"""CRPS and CRPS_sum (paper §5.2, eq. 5).

    CRPS(F, x) = integral_R (F(z) - I{x <= z})^2 dz                      (eq. 5)

CRPS is a proper scoring function, so it is minimized when the predictive
distribution equals the data distribution. With n samples X_i ~ F it is computed
directly from the samples through the energy form

    CRPS = E|X - y| - 0.5 * E|X - X'|

which is exact (not an approximation of the integral) and is the estimator used
everywhere in this file. E|X - X'| is evaluated with the sorted-sample identity

    E|X - X'| = (2 / n^2) * sum_i (2i - n + 1) * x_(i)      (i = 0 .. n-1)

so the cost is O(n log n) per series and time step instead of the O(n^2)
pairwise matrix -- that matters at D = 963 with n = 100. The paper uses n = 100.

CRPS_sum, and the ORDER OF OPERATIONS the paper spells out:

    "CRPS_sum is obtained by first summing across the D time-series -- both for
     the ground-truth data, and sampled data (yielding F_hat_sum(t) for each
     time point). The results are then averaged over the prediction horizon"

    CRPS_sum = E_t [ CRPS( F_hat_sum(t), sum_i x_t^i ) ]

Sum over D first, then CRPS per time point, then mean over the horizon.

NORMALIZATION (this is what makes the numbers comparable with Table 1): the
published CRPS_sum values -- 0.0064 for LSTM-Real-NVP on Exchange -- are scale
free. They are GluonTS `mean_wQuantileLoss` on the sum-aggregated series, i.e.
divided by sum_t |sum_i x_t^i|. A raw CRPS_sum in the original units cannot be
compared with them at all. Both are returned here:

    crps_sum(..., normalize=True)   -> comparable with Table 1   (default)
    crps_sum(..., normalize=False)  -> original units
    quantile_loss_sum(...)          -> the official 19-quantile evaluator value

Marginal CRPS (Appendix B.2) is the average CRPS over the D dimensions and the
predicted time steps -- a different metric, and the paper's models do not
uniformly win on it (Table 4).

Acceptance criterion (M5): a deliberately bad forecast (samples from N(0,1), no
model) must score much worse than the trained model. If the two are close, the
metric is broken, not the model.

Reference CRPS_sum for Exchange (Table 1): LSTM-Real-NVP 0.0064, LSTM-MAF 0.005.
"""

import torch

# Grid used by the authors' official MultivariateEvaluator notebook.
DEFAULT_QUANTILES = tuple(i / 20.0 for i in range(1, 20))


def crps_elementwise(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Energy-form CRPS per element.

    samples [n, *shape], target [*shape] -> [*shape].
    """
    try:
        if samples.dim() != target.dim() + 1:
            raise ValueError(
                f"samples {tuple(samples.shape)} must be target "
                f"{tuple(target.shape)} with a leading sample dimension"
            )
        samples = samples.to(torch.float64)
        target = target.to(torch.float64)
        n = samples.shape[0]

        term1 = (samples - target.unsqueeze(0)).abs().mean(dim=0)

        sorted_samples, _ = torch.sort(samples, dim=0)
        idx = torch.arange(n, device=samples.device, dtype=samples.dtype)
        weights = (2.0 * idx - n + 1.0).reshape(-1, *([1] * target.dim()))
        term2 = (2.0 / (n * n)) * (weights * sorted_samples).sum(dim=0)

        return term1 - 0.5 * term2
    except Exception as e:
        raise Exception(f"metrics - crps_elementwise - Execution error: {e}") from e


def crps_from_samples(samples: torch.Tensor, target) -> float:
    """CRPS of a single univariate predictive sample set. samples [n]."""
    try:
        samples = torch.as_tensor(samples).reshape(-1)
        target_t = torch.as_tensor(target, dtype=samples.dtype).reshape(())
        return float(crps_elementwise(samples, target_t).item())
    except Exception as e:
        raise Exception(f"metrics - crps_from_samples - Execution error: {e}") from e


def crps(samples: torch.Tensor, target: torch.Tensor,
         normalize: bool = True) -> float:
    """Marginal CRPS (Appendix B.2): mean over the D series and the horizon.

    samples [n, D, pred_len], target [D, pred_len].
    """
    try:
        _check_shapes(samples, target)
        values = crps_elementwise(samples, target)          # [D, pred_len]
        total = float(values.sum().item())
        if not normalize:
            return total / values.numel()
        denom = float(target.abs().to(torch.float64).sum().item())
        if denom <= 0:
            return total / values.numel()
        return total / denom
    except Exception as e:
        raise Exception(f"metrics - crps - Execution error: {e}") from e


def crps_sum(samples: torch.Tensor, target: torch.Tensor,
             normalize: bool = True) -> float:
    """CRPS_sum (§5.2): sum over D, CRPS per time point, mean over the horizon."""
    try:
        _check_shapes(samples, target)
        samples_sum = samples.sum(dim=1)                    # [n, pred_len]
        target_sum = target.sum(dim=0)                      # [pred_len]

        values = crps_elementwise(samples_sum, target_sum)  # [pred_len]
        total = float(values.sum().item())
        if not normalize:
            return total / values.numel()
        denom = float(target_sum.abs().to(torch.float64).sum().item())
        if denom <= 0:
            return total / values.numel()
        return total / denom
    except Exception as e:
        raise Exception(f"metrics - crps_sum - Execution error: {e}") from e


def quantile_loss_sum(samples: torch.Tensor, target: torch.Tensor,
                      quantiles=DEFAULT_QUANTILES) -> float:
    """GluonTS-style mean weighted quantile loss on the sum-aggregated series.

    Independent cross-check of `crps_sum(normalize=True)`: both estimate the same
    normalized quantity, one through the energy form and one through the pinball
    quadrature, so they should agree to a few percent.
    """
    try:
        _check_shapes(samples, target)
        numerator, denominator = quantile_loss_sum_parts(
            samples, target, quantiles=quantiles
        )
        return numerator / denominator if denominator > 0 else float("nan")
    except Exception as e:
        raise Exception(f"metrics - quantile_loss_sum - Execution error: {e}") from e


def quantile_loss_sum_parts(samples: torch.Tensor, target: torch.Tensor,
                            quantiles=DEFAULT_QUANTILES) -> tuple[float, float]:
    """Additive numerator/denominator for GluonTS mean_wQuantileLoss."""
    try:
        _check_shapes(samples, target)
        samples_sum = samples.sum(dim=1).to(torch.float64)
        target_sum = target.sum(dim=0).to(torch.float64)
        denominator = float(target_sum.abs().sum().item())
        numerators = []
        for q in quantiles:
            q_hat = torch.quantile(samples_sum, float(q), dim=0)
            error = target_sum - q_hat
            pinball = torch.maximum(float(q) * error,
                                    (float(q) - 1.0) * error)
            numerators.append(2.0 * float(pinball.sum().item()))
        return sum(numerators) / len(numerators), denominator
    except Exception as e:
        raise Exception(
            f"metrics - quantile_loss_sum_parts - Execution error: {e}"
        ) from e


def quantile_loss_parts(samples: torch.Tensor, target: torch.Tensor,
                        quantiles=DEFAULT_QUANTILES) -> tuple[float, float]:
    """Additive numerator/denominator for marginal mean_wQuantileLoss."""
    try:
        _check_shapes(samples, target)
        samples = samples.to(torch.float64)
        target = target.to(torch.float64)
        denominator = float(target.abs().sum().item())
        numerators = []
        for q in quantiles:
            q_hat = torch.quantile(samples, float(q), dim=0)
            error = target - q_hat
            pinball = torch.maximum(float(q) * error,
                                    (float(q) - 1.0) * error)
            numerators.append(2.0 * float(pinball.sum().item()))
        return sum(numerators) / len(numerators), denominator
    except Exception as e:
        raise Exception(
            f"metrics - quantile_loss_parts - Execution error: {e}"
        ) from e


def mse(samples: torch.Tensor, target: torch.Tensor,
        observed: torch.Tensor | None = None) -> float:
    """Appendix B.3: MSE of the sample mean, over all D and the horizon."""
    try:
        _check_shapes(samples, target)
        mean_pred = samples.mean(dim=0)
        if observed is None:
            return float(torch.nn.functional.mse_loss(mean_pred, target).item())
        weights = observed.to(dtype=mean_pred.dtype)
        squared = (mean_pred - target).square() * weights
        return float((squared.sum() / weights.sum().clamp_min(1.0)).item())
    except Exception as e:
        raise Exception(f"metrics - mse - Execution error: {e}") from e


def nd(samples: torch.Tensor, target: torch.Tensor) -> float:
    """Normalized deviation of the median forecast -- cheap extra diagnostic."""
    try:
        _check_shapes(samples, target)
        median = samples.median(dim=0).values
        denom = target.abs().sum()
        if float(denom.item()) <= 0:
            return float("nan")
        return float(((median - target).abs().sum() / denom).item())
    except Exception as e:
        raise Exception(f"metrics - nd - Execution error: {e}") from e


def compute_metrics(samples: torch.Tensor, target: torch.Tensor,
                    observed: torch.Tensor | None = None) -> dict:
    """All metrics for one test window, ready to be written to JSON."""
    try:
        # Keep additive numerators/denominators as well as the convenient
        # per-window ratios.  GluonTS's mean_wQuantileLoss (the Table-1
        # convention) aggregates ALL forecast points first and divides once:
        #
        #     sum_window numerator / sum_window denominator
        #
        # Averaging already-normalized window ratios is wrong and is especially
        # destructive for Taxi: the aggregate target is nearly zero in a couple
        # of windows, so an otherwise ordinary forecast receives a ratio of 46
        # or 110 and dominates the arithmetic mean.  The additive fields let
        # scripts/evaluate.py perform the published weighted aggregation.
        _check_shapes(samples, target)
        if observed is None:
            observed = torch.ones_like(target)
        if observed.shape != target.shape:
            raise ValueError(
                f"observed {tuple(observed.shape)} must match target "
                f"{tuple(target.shape)}"
            )
        observed = observed.to(dtype=target.dtype)
        target = target * observed
        samples = samples * observed.unsqueeze(0)

        samples_sum = samples.sum(dim=1)
        target_sum = target.sum(dim=0).to(torch.float64)
        crps_sum_values = crps_elementwise(samples_sum, target_sum)
        energy_sum_numerator = float(crps_sum_values.sum().item())
        crps_sum_denominator = float(target_sum.abs().sum().item())
        crps_sum_numerator, _ = quantile_loss_sum_parts(samples, target)

        marginal_values = crps_elementwise(samples, target)
        energy_numerator = float(marginal_values.sum().item())
        crps_numerator, crps_denominator = quantile_loss_parts(samples, target)

        median = samples.median(dim=0).values
        nd_numerator = float((median - target).abs().sum().item())

        return {
            "crps_sum": (crps_sum_numerator / crps_sum_denominator
                          if crps_sum_denominator > 0
                          else float("nan")),
            "crps_sum_raw": crps_sum_numerator / target.shape[-1],
            "crps_sum_quantile": (crps_sum_numerator / crps_sum_denominator
                                   if crps_sum_denominator > 0 else float("nan")),
            "crps_sum_energy": (energy_sum_numerator / crps_sum_denominator
                                if crps_sum_denominator > 0 else float("nan")),
            "crps_sum_energy_raw": energy_sum_numerator / target.shape[-1],
            "crps": (crps_numerator / crps_denominator
                      if crps_denominator > 0
                      else float("nan")),
            "crps_raw": crps_numerator / target.numel(),
            "crps_energy": (energy_numerator / crps_denominator
                            if crps_denominator > 0 else float("nan")),
            "crps_energy_raw": energy_numerator / target.numel(),
            "mse": mse(samples, target, observed=observed),
            "nd": (nd_numerator / crps_denominator
                   if crps_denominator > 0 else float("nan")),
            # Additive bookkeeping used for aggregation across rolling windows.
            "_crps_sum_numerator": crps_sum_numerator,
            "_crps_sum_denominator": crps_sum_denominator,
            "_crps_sum_energy_numerator": energy_sum_numerator,
            "_crps_numerator": crps_numerator,
            "_crps_denominator": crps_denominator,
            "_crps_energy_numerator": energy_numerator,
            "_nd_numerator": nd_numerator,
        }
    except Exception as e:
        raise Exception(f"metrics - compute_metrics - Execution error: {e}") from e


def _check_shapes(samples: torch.Tensor, target: torch.Tensor) -> None:
    if samples.dim() != 3:
        raise ValueError(
            f"samples must be [n, D, pred_len], got {tuple(samples.shape)}"
        )
    if target.dim() != 2:
        raise ValueError(
            f"target must be [D, pred_len], got {tuple(target.shape)}"
        )
    if samples.shape[1:] != target.shape:
        raise ValueError(
            f"samples {tuple(samples.shape)} and target {tuple(target.shape)} "
            f"disagree on D or pred_len"
        )
