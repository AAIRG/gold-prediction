"""Metric tests (roadmap M5).

The acceptance criterion the paper's numbers depend on: a deliberately bad
forecast must score much worse than a good one, and CRPS_sum must be normalized
the way Table 1 is -- otherwise 0.0064 is not a target, just a number.
"""

import torch

from flowforecast.evaluation.metrics import (compute_metrics, crps,
                                             crps_elementwise, crps_from_samples,
                                             crps_sum, mse, quantile_loss_sum,
                                             quantile_loss_sum_parts)


def _pairwise_crps(samples: torch.Tensor, target: float) -> float:
    """O(n^2) reference implementation of E|X-y| - 0.5 E|X-X'|."""
    term1 = (samples - target).abs().mean()
    term2 = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs().mean()
    return float((term1 - 0.5 * term2).item())


def test_point_mass_crps_is_the_absolute_error():
    samples = torch.full((50,), 3.0)
    assert abs(crps_from_samples(samples, 5.0) - 2.0) < 1e-6


def test_sorted_identity_matches_the_pairwise_form():
    torch.manual_seed(0)
    samples = torch.randn(64, dtype=torch.float64)
    for target in (-1.0, 0.0, 0.7):
        fast = crps_from_samples(samples, target)
        slow = _pairwise_crps(samples, target)
        assert abs(fast - slow) < 1e-9


def test_elementwise_shapes():
    samples = torch.randn(20, 4, 6)
    target = torch.randn(4, 6)
    values = crps_elementwise(samples, target)
    assert values.shape == (4, 6)
    assert torch.all(values >= 0)


def test_crps_sum_uses_the_right_order_of_operations():
    """Sum over D first, then CRPS per time point, then average (§5.2)."""
    torch.manual_seed(0)
    samples = torch.randn(32, 3, 4) + 2.0
    target = torch.randn(3, 4) + 2.0

    manual = []
    samples_sum = samples.sum(dim=1)
    target_sum = target.sum(dim=0)
    for t in range(4):
        manual.append(crps_from_samples(samples_sum[:, t], float(target_sum[t])))
    expected_raw = sum(manual) / len(manual)
    assert abs(crps_sum(samples, target, normalize=False) - expected_raw) < 1e-8

    denom = float(target_sum.abs().sum())
    expected_norm = sum(manual) / denom
    assert abs(crps_sum(samples, target, normalize=True) - expected_norm) < 1e-8


def test_normalization_is_scale_free():
    torch.manual_seed(0)
    target = torch.rand(5, 8) + 1.0
    samples = target.unsqueeze(0) + 0.1 * torch.randn(64, 5, 8)
    a = crps_sum(samples, target, normalize=True)
    b = crps_sum(samples * 1000.0, target * 1000.0, normalize=True)
    assert abs(a - b) / a < 1e-6
    # ... while the raw value scales with the data
    raw_a = crps_sum(samples, target, normalize=False)
    raw_b = crps_sum(samples * 1000.0, target * 1000.0, normalize=False)
    assert raw_b > 100 * raw_a


def test_quantile_loss_parts_are_additive():
    torch.manual_seed(0)
    target = torch.rand(6, 10) + 2.0
    samples = target.unsqueeze(0) + 0.2 * torch.randn(200, 6, 10)
    numerator, denominator = quantile_loss_sum_parts(samples, target)
    assert denominator > 0
    assert abs(quantile_loss_sum(samples, target)
               - numerator / denominator) < 1e-12


def test_bad_forecast_scores_much_worse_than_a_good_one():
    """M5 acceptance test: N(0,1) noise vs a sharp forecast."""
    torch.manual_seed(0)
    target = torch.rand(4, 12) * 3.0 + 5.0
    good = target.unsqueeze(0) + 0.05 * torch.randn(100, 4, 12)
    bad = torch.randn(100, 4, 12)

    assert crps_sum(bad, target) > 10 * crps_sum(good, target)
    assert crps(bad, target) > 10 * crps(good, target)
    assert mse(bad, target) > 10 * mse(good, target)


def test_compute_metrics_returns_every_key():
    torch.manual_seed(0)
    target = torch.rand(3, 5) + 1.0
    samples = target.unsqueeze(0) + 0.1 * torch.randn(32, 3, 5)
    metrics = compute_metrics(samples, target)
    for key in ("crps_sum", "crps_sum_raw", "crps_sum_quantile",
                "crps_sum_energy", "crps_sum_energy_raw", "crps",
                "crps_raw", "crps_energy", "crps_energy_raw", "mse", "nd"):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_rolling_crps_sum_must_be_ratio_of_global_sums():
    """A near-zero window must not dominate the published aggregate metric."""
    from scripts.evaluate import _summarize

    # Same raw numerator in both windows, but the first denominator is tiny.
    # Arithmetic mean of ratios is 50.05; Table-1 aggregation is 2/10.01.
    windows = [
        {"crps_sum": 100.0, "crps_sum_quantile": 100.0,
         "crps": 1.0, "mse": 1.0, "nd": 1.0,
         "_crps_sum_numerator": 1.0, "_crps_sum_denominator": 0.01,
         "_crps_numerator": 1.0, "_crps_denominator": 1.0,
         "_nd_numerator": 1.0},
        {"crps_sum": 0.1, "crps_sum_quantile": 0.1,
         "crps": 1.0, "mse": 1.0, "nd": 1.0,
         "_crps_sum_numerator": 1.0, "_crps_sum_denominator": 10.0,
         "_crps_numerator": 1.0, "_crps_denominator": 1.0,
         "_nd_numerator": 1.0},
    ]
    result = _summarize(windows)
    assert abs(result["crps_sum"]["mean"] - 2.0 / 10.01) < 1e-12
    assert result["crps_sum"]["mean"] < 1.0
