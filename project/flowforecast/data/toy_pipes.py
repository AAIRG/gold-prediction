"""Simulated flow of liquid in a system of pipes (paper §5.1, Figures 3 and 4).

This is the paper's own sanity check: does the model learn ground-truth
dependency structure, both across series (space) and across time?

Generative process (§5.1, verbatim):

    S_0 = X + 3,          X ~ Gamma(1, 0.2)     # shape/scale parameterization
    V_1, V_2 ~ iid Beta(0.5, 0.5)
    S_i = V_i / (V_1 + V_2) * S_0 + eps_i       for i in {1, 2}
    S_3 = S_1 + S_2 + eps_3
    eps_* ~ N(0, 0.1)

The observation vector is 6-dimensional: (v1, v2, s0, s1, s2, s3) -- matching the
axis labels of Figure 4.

Temporal structure (Figure 4 caption): the cross-covariance between consecutive
time points "reflects the true flow of liquid in the system from S_0 at time t to
S_1 and S_2 at time t+1, on to S_3 at time t+2", so liquid propagates with a
one-step lag per stage.

Two things the paper checks, and our acceptance criteria (roadmap M6):
    1. The model's covariance at a fixed time point must reproduce the data's
       covariance, including the sign of Cov(S_1, S_2) (Figure 4 left). Note that
       with Beta(0.5, 0.5) splitting an almost constant S_0 that empirical
       covariance is NEGATIVE -- the split dominates the shared source -- so the
       criterion is "matches the data", not "is positive".
    2. Cross-covariance between consecutive time points must show the
       S_0 -> {S_1, S_2} -> S_3 propagation pattern with positive entries
       (Figure 4 right).

Implementation notes: the first two time steps have no upstream history, so the
simulation burns them in and drops them instead of emitting zeros -- leaving them
in puts two artificial all-zero rows into every covariance estimate. Randomness
goes through a local torch.Generator, so calling this never disturbs the global
seed of a training run.
"""

import torch

PIPE_LABELS = ["v1", "v2", "s0", "s1", "s2", "s3"]

# Row indices inside the observation vector, for the assertions of M6.
V1, V2, S0, S1, S2, S3 = range(6)


def simulate_pipes(num_timesteps: int, seed: int | None = None,
                   noise_std: float = 0.1) -> torch.Tensor:
    """[6, num_timesteps] observations of the §5.1 pipe system."""
    try:
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))

        burn_in = 2
        n = int(num_timesteps) + burn_in

        # Gamma(shape=1, scale=0.2) == Gamma(concentration=1, rate=5)
        uniform = torch.rand(n, generator=generator).clamp_min(1e-12)
        X = -torch.log(uniform) * 0.2                      # Exp(1) * scale
        S_0 = X + 3.0

        V_1 = _sample_beta(n, 0.5, 0.5, generator)
        V_2 = _sample_beta(n, 0.5, 0.5, generator)

        eps = torch.randn(4, n, generator=generator) * noise_std

        share = (V_1 / (V_1 + V_2).clamp_min(1e-12))
        S_1 = torch.zeros(n)
        S_2 = torch.zeros(n)
        S_3 = torch.zeros(n)

        # One-step lag per stage: S_0 at t feeds S_1, S_2 at t+1, which feed S_3
        # at t+2.
        S_1[1:] = share[1:] * S_0[:-1] + eps[1, 1:]
        S_2[1:] = (1.0 - share[1:]) * S_0[:-1] + eps[2, 1:]
        S_3[2:] = S_1[1:-1] + S_2[1:-1] + eps[3, 2:]

        series = torch.stack([V_1, V_2, S_0, S_1, S_2, S_3], dim=0)
        return series[:, burn_in:].contiguous()
    except Exception as e:
        raise Exception(f"toy_pipes - simulate_pipes - Execution error: {e}") from e


def _sample_beta(n: int, alpha: float, beta: float,
                 generator: torch.Generator | None) -> torch.Tensor:
    """Beta(a, b) from two Gammas, honouring an explicit generator."""
    try:
        # Gamma sampling with a generator is not exposed by torch.distributions,
        # so use the Johnk / ratio-of-gammas identity via inverse-CDF-free method:
        # Beta(0.5, 0.5) is the arcsine distribution, whose CDF is analytic; for
        # the general case fall back to torch.distributions (global RNG).
        if abs(alpha - 0.5) < 1e-12 and abs(beta - 0.5) < 1e-12:
            u = torch.rand(n, generator=generator)
            return torch.sin(0.5 * torch.pi * u) ** 2       # arcsine inverse CDF
        return torch.distributions.Beta(alpha, beta).sample((n,))
    except Exception as e:
        raise Exception(f"toy_pipes - _sample_beta - Execution error: {e}") from e


def empirical_covariance(series: torch.Tensor) -> torch.Tensor:
    """[D, D] covariance of x_t with itself (Figure 4, left)."""
    try:
        series = series.to(torch.float64)
        centered = series - series.mean(dim=1, keepdim=True)
        return (centered @ centered.t() / max(1, series.shape[1] - 1)).float()
    except Exception as e:
        raise Exception(f"toy_pipes - empirical_covariance - Execution error: {e}") from e


def empirical_cross_covariance(series: torch.Tensor) -> torch.Tensor:
    """[D, D] covariance of x_t (rows) with x_{t+1} (columns), Figure 4 right."""
    try:
        series = series.to(torch.float64)
        x_t = series[:, :-1]
        x_next = series[:, 1:]
        centered_t = x_t - x_t.mean(dim=1, keepdim=True)
        centered_next = x_next - x_next.mean(dim=1, keepdim=True)
        return (centered_t @ centered_next.t()
                / max(1, x_t.shape[1] - 1)).float()
    except Exception as e:
        raise Exception(
            f"toy_pipes - empirical_cross_covariance - Execution error: {e}"
        ) from e


def check_pipe_structure(series: torch.Tensor, tolerance: float = 0.0) -> dict:
    """The M6 structure numbers, instead of eyeballed heatmaps.

    `propagation_visible` is the robust criterion: with S_0 feeding S_1 and S_2
    one step later, and those feeding S_3 one step after that, all four
    cross-covariance entries S_0->S_1, S_0->S_2, S_1->S_3, S_2->S_3 must be
    positive (their expected size is E[V_1/(V_1+V_2)] * Var(S_0)).

    `cov_s1_s2` is reported but NOT asserted to be positive: with Beta(0.5, 0.5)
    splitting an almost constant source, the split dominates and the empirical
    sign is negative. The experiment's real question is whether the fitted model
    reproduces the data's matrices, which is what scripts/pipes_experiment.py
    compares.
    """
    try:
        cov = empirical_covariance(series)
        cross = empirical_cross_covariance(series)
        return {
            "cov_s1_s2": float(cov[S1, S2].item()),
            "cov_s0_s3": float(cov[S0, S3].item()),
            "cross_s0_s1": float(cross[S0, S1].item()),
            "cross_s0_s2": float(cross[S0, S2].item()),
            "cross_s1_s3": float(cross[S1, S3].item()),
            "cross_s2_s3": float(cross[S2, S3].item()),
            "propagation_visible": bool(
                cross[S0, S1].item() > tolerance
                and cross[S0, S2].item() > tolerance
                and cross[S1, S3].item() > tolerance
                and cross[S2, S3].item() > tolerance
            ),
        }
    except Exception as e:
        raise Exception(f"toy_pipes - check_pipe_structure - Execution error: {e}") from e
