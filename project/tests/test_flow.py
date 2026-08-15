"""Unit tests for the flow blocks. These gate milestones M1, M2 and M7.

Run with:  python -m pytest tests -v          (from the project directory)

M1 -- invertibility and log-det:
    ConditionalCouplingLayer, BatchNormFlow (eval mode), Permutation, and the
    whole ConditionalFlow stack must round-trip, and the analytic log-det must
    agree with torch.autograd.functional.jacobian for D = 4.

M2 -- the conditioning is actually used:
    a flow that silently ignores h still shows a decreasing loss, so the only
    reliable check is behavioural: train on a two-component mixture where h is
    the component label, then verify samples follow h.

M7 -- MAF:
    same invertibility / log-det tests with the sequential inverse, plus
    autoregressivity (perturbing x^j must not change y^i for i <= j) and the
    reversal of MADE input degrees between stacked MAF blocks.
"""

import pytest
import torch

from flowforecast.flows.batchnorm import BatchNormFlow
from flowforecast.flows.coupling import ConditionalCouplingLayer
from flowforecast.flows.flow import ConditionalFlow, Permutation
from flowforecast.flows.made import MADE
from flowforecast.flows.maf import ConditionalMAFLayer

BATCH, D, COND = 32, 4, 3


def _inputs(batch=BATCH, d=D, cond=COND, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, d), torch.randn(batch, cond)


def _randomize_made_output(made: MADE) -> None:
    output_layer = made.layers[-1]
    torch.nn.init.normal_(output_layer.weight, mean=0.0, std=0.05)
    torch.nn.init.normal_(output_layer.bias, mean=0.0, std=0.05)
    if output_layer.cond_weight is not None:
        torch.nn.init.normal_(output_layer.cond_weight, mean=0.0, std=0.05)


# --------------------------------------------------------------------------- M1
def test_coupling_invertibility():
    x, h = _inputs()
    block = ConditionalCouplingLayer(features=D, cond_size=COND)
    y, _ = block.forward(x, h)
    x_rec = block.inverse(y, h)
    assert (x - x_rec).abs().max() < 1e-4


def test_coupling_log_det_matches_autograd():
    x, h = _inputs()
    block = ConditionalCouplingLayer(features=D, cond_size=COND)
    x_single, h_single = x[0:1], h[0:1]
    _, log_det = block.forward(x_single, h_single)

    def forward_fn(v):
        return block.forward(v, h_single)[0].squeeze(0)

    jacobian = torch.autograd.functional.jacobian(forward_fn, x_single).squeeze()
    assert torch.allclose(torch.logdet(jacobian).abs(), log_det.abs(),
                          rtol=1e-3, atol=1e-3)


def test_batchnorm_invertibility_in_eval_mode():
    x, _ = _inputs()
    block = BatchNormFlow(features=D)

    block.train()
    for _ in range(20):                    # populate the running statistics
        block(torch.randn(BATCH, D) * 2.0 + 1.0)

    block.eval()
    y, log_det = block(x)
    x_rec = block.inverse(y)
    assert (x - x_rec).abs().max() < 1e-4
    assert log_det.shape == (BATCH,)


def test_batchnorm_log_det_matches_autograd_in_eval_mode():
    x, _ = _inputs()
    block = BatchNormFlow(features=D)
    block.train()
    for _ in range(20):
        block(torch.randn(BATCH, D) * 1.5 - 0.5)
    block.eval()

    x_single = x[0:1]
    _, log_det = block(x_single)

    jacobian = torch.autograd.functional.jacobian(
        lambda v: block(v)[0].squeeze(0), x_single).squeeze()
    assert torch.allclose(torch.logdet(jacobian).abs(), log_det[0].abs(),
                          rtol=1e-4, atol=1e-5)


def test_permutation_is_a_volume_preserving_bijection():
    x, _ = _inputs()
    block = Permutation(features=D)
    y, log_det = block(x)
    assert torch.allclose(log_det, torch.zeros(BATCH))
    assert torch.equal(block.inverse(y), x)


@pytest.mark.parametrize("block_type", ["realnvp", "maf"])
def test_full_stack_round_trip_and_shapes(block_type):
    x, h = _inputs()
    flow = ConditionalFlow(features=D, cond_size=COND, num_blocks=3, hidden=16,
                           block_type=block_type)
    if block_type == "maf":
        for block in flow.blocks:
            if isinstance(block, ConditionalMAFLayer):
                _randomize_made_output(block.made)
    flow.train()
    for _ in range(10):                    # warm up the BatchNorm buffers
        flow.log_prob(torch.randn(BATCH, D), torch.randn(BATCH, COND))

    flow.eval()
    z, log_det = flow.forward(x, h)
    assert z.shape == x.shape
    assert log_det.shape == (BATCH,)

    # Round trip through every block, in reverse.
    recovered = z
    for block in reversed(flow.blocks):
        recovered = block.inverse(recovered, h)
    assert (recovered - x).abs().max() < 1e-3

    log_prob = flow.log_prob(x, h)
    assert log_prob.shape == (BATCH,)
    assert torch.isfinite(log_prob).all()

    samples = flow.sample(h)
    assert samples.shape == (BATCH, D)
    assert torch.isfinite(samples).all()


def test_every_dimension_is_transformed_somewhere_in_the_stack():
    """A dimension that no coupling layer ever touches keeps a valid log_det
    while its samples stay pure noise -- alternate the mask, roadmap section 7."""
    flow = ConditionalFlow(features=D, cond_size=COND, num_blocks=2, hidden=8,
                           block_type="realnvp")
    couplings = [b for b in flow.blocks if isinstance(b, ConditionalCouplingLayer)]
    assert len(couplings) == 2
    transformed = sum((1.0 - c.mask) for c in couplings)
    assert (transformed > 0).all()


def test_maf_stack_reverses_autoregressive_degrees():
    flow = ConditionalFlow(features=D, cond_size=COND, num_blocks=3, hidden=8,
                           block_type="maf")
    mafs = [b for b in flow.blocks if isinstance(b, ConditionalMAFLayer)]
    assert len(mafs) == 3
    assert torch.equal(mafs[1].input_degrees, mafs[0].input_degrees.flip(0))
    assert torch.equal(mafs[2].input_degrees, mafs[1].input_degrees.flip(0))


# --------------------------------------------------------------------------- M7
def test_maf_invertibility():
    x, h = _inputs()
    block = ConditionalMAFLayer(features=D, cond_size=COND, hidden=32)
    _randomize_made_output(block.made)
    y, _ = block.forward(x, h)
    x_rec = block.inverse(y, h)
    assert (x - x_rec).abs().max() < 1e-4


def test_maf_soft_bound_prevents_inverse_scale_overflow(monkeypatch):
    """The observed raw alpha=126 failure must remain finite and invertible."""
    monkeypatch.setenv("FLOWFORECAST_MAF_DIAGNOSTICS", "1")
    block = ConditionalMAFLayer(
        features=D, cond_size=COND, hidden=8, log_scale_bound=5.0
    )
    x, h = _inputs()

    def unstable_made(x, condition):
        del condition
        return torch.zeros_like(x), torch.full_like(x, 126.054)

    monkeypatch.setattr(block.made, "forward", unstable_made)
    y, log_det = block.forward(x, h)
    recovered = block.inverse(y, h)

    assert torch.isfinite(y).all()
    assert torch.isfinite(log_det).all()
    assert torch.isfinite(recovered).all()
    assert torch.allclose(recovered, x, rtol=1e-4, atol=1e-4)
    diagnostics = block.last_inverse_diagnostics
    assert diagnostics["raw_alpha_abs_max"] == pytest.approx(126.054)
    assert diagnostics["alpha_abs_max"] <= 5.0
    assert diagnostics["scale_max"] == pytest.approx(
        torch.exp(torch.tensor(5.0)).item()
    )


def test_maf_soft_bound_is_near_identity_around_zero():
    block = ConditionalMAFLayer(features=D, cond_size=COND, hidden=8,
                                log_scale_bound=5.0)
    raw = torch.tensor([-1e-3, 0.0, 1e-3, 126.054])
    bounded = block._bound_log_scale(raw)

    assert torch.allclose(bounded[:3], raw[:3], atol=1e-8)
    assert bounded.abs().max() <= 5.0


def test_maf_default_matches_official_unbounded_parameterization():
    block = ConditionalMAFLayer(features=D, cond_size=COND, hidden=8)
    raw = torch.tensor([-7.0, 0.0, 9.0])

    assert block.log_scale_bound == 0.0
    assert torch.equal(block._bound_log_scale(raw), raw)


def test_made_uses_standard_nonzero_initialization():
    x, h = _inputs()
    block = ConditionalMAFLayer(features=D, cond_size=COND, hidden=16)

    raw_mu, raw_alpha = block.made(x, h)

    assert torch.count_nonzero(raw_mu) > 0
    assert torch.count_nonzero(raw_alpha) > 0


def test_maf_log_det_matches_autograd():
    x, h = _inputs()
    block = ConditionalMAFLayer(features=D, cond_size=COND, hidden=32)
    _randomize_made_output(block.made)
    x_single, h_single = x[0:1], h[0:1]
    _, log_det = block.forward(x_single, h_single)

    jacobian = torch.autograd.functional.jacobian(
        lambda v: block.forward(v, h_single)[0].squeeze(0), x_single).squeeze()
    assert torch.allclose(torch.logdet(jacobian).abs(), log_det.abs(),
                          rtol=1e-3, atol=1e-3)


def test_made_is_autoregressive():
    """Output i must not react to input j >= i (paper §2.1, MADE requirement)."""
    torch.manual_seed(0)
    made = MADE(features=6, cond_size=COND, hidden=64)
    _randomize_made_output(made)
    x = torch.randn(1, 6)
    h = torch.randn(1, COND)
    mu, alpha = made(x, h)

    for j in range(6):
        perturbed = x.clone()
        perturbed[0, j] += 5.0
        mu_p, alpha_p = made(perturbed, h)
        # everything up to and including j must be unchanged
        assert torch.allclose(mu[0, :j + 1], mu_p[0, :j + 1], atol=1e-6)
        assert torch.allclose(alpha[0, :j + 1], alpha_p[0, :j + 1], atol=1e-6)


def test_maf_forward_is_autoregressive():
    torch.manual_seed(0)
    block = ConditionalMAFLayer(features=5, cond_size=COND, hidden=32)
    _randomize_made_output(block.made)
    x = torch.randn(1, 5)
    h = torch.randn(1, COND)
    y, _ = block.forward(x, h)

    for j in range(5):
        perturbed = x.clone()
        perturbed[0, j] += 3.0
        y_p, _ = block.forward(perturbed, h)
        assert torch.allclose(y[0, :j], y_p[0, :j], atol=1e-6)


# --------------------------------------------------------------------------- M2
def test_conditioning_is_actually_used():
    """Two-component mixture with h as the label: samples must follow h.

    This is the only test that catches a flow which ignores its conditioning --
    the loss curve looks fine either way.
    """
    torch.manual_seed(0)
    features, cond = 2, 1
    flow = ConditionalFlow(features=features, cond_size=cond, num_blocks=4,
                           hidden=32, block_type="realnvp")
    optimizer = torch.optim.Adam(flow.parameters(), lr=5e-3)

    def batch(n=512):
        label = torch.randint(0, 2, (n, 1)).float()
        centre = torch.where(label > 0.5, 4.0, -4.0)
        x = centre + 0.3 * torch.randn(n, features)
        return x, label

    flow.train()
    for _ in range(400):
        x, label = batch()
        loss = -flow.log_prob(x, label).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        optimizer.step()

    flow.eval()
    with torch.no_grad():
        positive = flow.sample(torch.ones(256, 1))
        negative = flow.sample(torch.zeros(256, 1))

    assert positive.mean() > 2.0, f"samples for h=1 sit at {positive.mean():.2f}"
    assert negative.mean() < -2.0, f"samples for h=0 sit at {negative.mean():.2f}"


def test_density_of_a_known_gaussian():
    """M1: fit a fixed-covariance Gaussian and recover its covariance."""
    torch.manual_seed(0)
    features = 2
    cov = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
    chol = torch.linalg.cholesky(cov)

    flow = ConditionalFlow(features=features, cond_size=1, num_blocks=4,
                           hidden=32, block_type="realnvp")
    optimizer = torch.optim.Adam(flow.parameters(), lr=5e-3)

    flow.train()
    for _ in range(400):
        x = torch.randn(512, features) @ chol.t()
        h = torch.zeros(512, 1)
        loss = -flow.log_prob(x, h).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    flow.eval()
    with torch.no_grad():
        samples = flow.sample(torch.zeros(8192, 1))
    empirical = torch.cov(samples.t())
    assert torch.allclose(empirical, cov, atol=0.25), f"got\n{empirical}"
