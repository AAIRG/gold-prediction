"""Batch Normalization as a bijection between coupling layers.

The paper (§2.1, last paragraph) is explicit about this:

    "In practice, Batch Normalization (Ioffe & Szegedy, 2015) is applied as a
     bijection to outputs of successive coupling layers to stabilize the
     training of normalizing flows. This bijection implements the
     normalization procedure using a weighted moving average of the layer's
     mean and standard deviation values, which has to be adapted to either
     training or inference regimes."

Figure 1 of the paper shows the blocks as: Coupling Layer 1 -> Batch Norm 1
-> ... -> Coupling Layer K -> Batch Norm K.

Forward (x -> y):
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    log|det J| = sum_j ( log|gamma_j| - 0.5 * log(var_j + eps) )

train mode uses the batch statistics and updates the running buffers; eval mode
uses the frozen running buffers. Getting that split wrong is a classic source of
"the loss was fine but sampling is garbage" (roadmap section 7): sampling always
runs in eval mode, so `inverse` uses the running statistics unconditionally.

`min_var` is OUR numerical guard and it is the single most important line in
this file. It is not in the paper, exactly like `scale_clamp: tanh` in
flows/coupling.py is not in the paper: both bound an otherwise unbounded
quantity. Here the unbounded quantity is the normalisation gain 1/sqrt(var).

Why it matters (this is a real failure we debugged, not a hypothetical): the
transform divides every dimension by its own standard deviation. If ONE
dimension is (near-)constant inside the minibatch, its var collapses toward 0
and the gain 1/sqrt(var + eps) saturates at 1/sqrt(eps) ~ 316 for eps = 1e-5.
That single dimension then:

  * destroys the density in eval mode: z = (x - mean) / sqrt(var + eps) becomes
    ~316, so z^2 / 2 contributes ~5e4 per dimension. Over D = 137 that is ~7e6,
    which is the order of the "train_nll -566 but val_nll +724337" explosion
    observed on Solar/MAF, and of the smaller "+426" one on Electricity.
  * destroys sampling in the opposite direction: `inverse` multiplies by
    sqrt(var + eps) ~ 3e-3, so that dimension's samples collapse onto `mean`.

That opposite-direction pair explains the most confusing symptom we saw: on
Exchange, the checkpoint with the BEST validation NLL had the WORST CRPS, in all
three flow variants. A checkpoint with a collapsed running_var scores a
fantastic density (the -0.5*log(var + eps) term in log_det is large and positive)
while producing garbage forecasts. Selecting on val_nll was therefore selecting
the worst forecaster.

It is dimension-collapse, not overfitting, and it is why the datasets with
zero-inflated / sparse cross-sections (Solar at night, count data) exploded
while dense Exchange only looked noisy.

HONESTY ABOUT THE REFERENCE IMPLEMENTATION (zalandoresearch/pytorch-ts,
pts/modules/flows.py, written by the paper's first author): it has NO variance
floor. Its only guard is the same `var + eps` with eps = 1e-5, i.e. it also
tolerates a gain of ~316. So `min_var` is stricter than the reference, and it is
tagged [ours] for that reason -- it is not "restoring" anything. Note also that
the old `clamp_min(1e-8)` in this file was a NO-OP: `var + eps` with eps = 1e-5
already dominated it, so removing it changed nothing by itself.

`momentum`, on the other hand, now matches the reference exactly. The reference
writes `running.mul_(0.9).add_(batch * 0.1)`, i.e. a new-batch weight of 0.1.
Our convention is `running.mul_(1 - momentum).add_(batch * momentum)`, so
momentum = 0.1 here IS the reference's momentum = 0.9, and it is the "weighted
moving average" §2.1 prescribes.

`min_var` bounds the gain at 1/sqrt(min_var) and is applied identically in train
mode, in eval mode and in `inverse`, so the block stays an exact bijection with
a matching log-determinant (tests/test_flow.py checks both). Set
`flow.bn_min_var: 0.0` for a run that is bit-for-bit faithful to the reference.

The signature is `forward(x, h=None)` so that every block in
flows/flow.py has the same interface; BatchNorm ignores the conditioning.
"""

import torch
import torch.nn as nn


class BatchNormFlow(nn.Module):
    conditional = False

    def __init__(self, features: int, momentum: float = 0.1, eps: float = 1e-5,
                 min_var: float = 0.0):
        super().__init__()
        try:
            self.features = features
            self.momentum = momentum
            self.eps = eps
            self.min_var = float(min_var)

            self.log_gamma = nn.Parameter(torch.zeros(features))
            self.beta = nn.Parameter(torch.zeros(features))

            self.register_buffer('running_mean', torch.zeros(features))
            self.register_buffer('running_var', torch.ones(features))

            # Last minibatch statistics, kept detached. The reference keeps the
            # same pair so that `inverse` can mirror `forward` while training.
            self._batch_mean: torch.Tensor | None = None
            self._batch_var: torch.Tensor | None = None
        except Exception as e:
            raise Exception(f"BatchNormFlow - __init__ - Init error: {e}") from e

    def _floor(self, var: torch.Tensor) -> torch.Tensor:
        """Bound the normalisation gain at 1/sqrt(min_var). See the docstring."""
        return var if self.min_var <= 0.0 else var.clamp_min(self.min_var)

    def _batch_statistics(self, x: torch.Tensor):
        """Mean/var over every leading axis, so [N, D] and [B, T, D] both work.

        The reference flattens with `view(-1, shape[-1])` and uses PyTorch's
        default unbiased variance estimate; we match both.
        """
        flat = x.reshape(-1, x.shape[-1])
        return flat.mean(dim=0), flat.var(dim=0, unbiased=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        try:
            if self.training:
                mean, var = self._batch_statistics(x)

                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(
                        self.momentum * mean.detach())
                    self.running_var.mul_(1 - self.momentum).add_(
                        self.momentum * var.detach())
                # Kept for `inverse`, which mirrors `forward` while training.
                self._batch_mean = mean.detach()
                self._batch_var = var.detach()
            else:
                mean = self.running_mean
                var = self.running_var

            var = self._floor(var)

            gamma = torch.exp(self.log_gamma)
            y = (x - mean) / torch.sqrt(var + self.eps) * gamma + self.beta

            log_det = torch.sum(self.log_gamma - 0.5 * torch.log(var + self.eps))
            log_det = log_det.expand(x.shape[0])

            return y, log_det
        except Exception as e:
            raise Exception(f"BatchNormFlow - forward - Execution error: {e}") from e

    def inverse(self, y: torch.Tensor, h: torch.Tensor | None = None):
        try:
            # Mirror `forward`: batch statistics while training, running
            # statistics in eval mode. Sampling runs in eval mode, so it takes
            # the running branch -- but keeping the pair symmetric means a
            # train-mode round trip is exact too, which is what the reference
            # implementation does and what tests/test_flow.py exercises.
            if self.training and self._batch_mean is not None:
                mean = self._batch_mean
                var = self._batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            var = self._floor(var)
            gamma = torch.exp(self.log_gamma)

            x = (y - self.beta) / gamma * torch.sqrt(var + self.eps) + mean
            return x
        except Exception as e:
            raise Exception(f"BatchNormFlow - inverse - Execution error: {e}") from e
