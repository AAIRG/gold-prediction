"""Conditional Real NVP affine coupling layer (paper §2.1, eq. 2).

Unconditional form:
    y^{1:d}   = x^{1:d}
    y^{d+1:D} = x^{d+1:D} * exp(s(x^{1:d})) + t(x^{1:d})

Conditional form (§2.1, "conditioning of the PDF p_X(x|h)"): h is concatenated
to the inputs of the scale and translation networks, which then map
R^{d+H} -> R^{D-d}:
    s(concat(x^{1:d}, h)),  t(concat(x^{1:d}, h))

Two facts from the paper worth remembering for the presentation:
    - This is allowed because s and t "don't need to be invertible".
    - Conditioning "does not change the log-determinant of the coupling
      layers given by (2)", i.e. conditioning is free w.r.t. the Jacobian.

The Jacobian is block-triangular, so (eq. 2):
    log|det(dy/dx)| = sum(s(x^{1:d}, h))

TODO (M1 unconditional, M2 conditional):
    - __init__(self, features: int, cond_size: int, hidden: int = 100,
               mask: torch.Tensor | None = None)
        * mask selects pass-through dims (x^{1:d}) vs transformed dims
          (x^{d+1:D}); ConditionalFlow alternates it across blocks.
        * s_net / t_net: linear feed-forward, hidden = 100, ELU activation
          (Appendix D.2). Input size features + cond_size.
        * IMPORTANT: bound the s output (s = tanh(s_raw), or clamp) before
          exp(). The paper does not mention this; it is required in practice
          (roadmap section 7 -- first thing to check on NaN).
    - forward(self, x, h) -> (y, log_det)      # density evaluation / training
    - inverse(self, y, h) -> x                  # sampling (M5)
        * x^{d+1:D} = (y^{d+1:D} - t(...)) * exp(-s(...))

Acceptance criteria (tests/test_flow.py):
    - max|inverse(forward(x)) - x| < 1e-5
    - analytic log_det matches torch.autograd.functional.jacobian for D=4
"""

import torch
import torch.nn as nn
import copy


class ConditionalCouplingLayer(nn.Module):
    conditional = True

    def __init__(self, features: int, cond_size: int, hidden: int = 100,
                 mask: torch.Tensor | None = None):
        super().__init__()
        try:
            self.features = features
            self.cond_size = cond_size

            if mask is None:
                mask = torch.zeros(features, dtype=torch.float32)
                mask[:features // 2] = 1.0
            self.register_buffer('mask', mask)

            in_features = features + cond_size

            self.s_net = nn.Sequential(
                nn.Linear(in_features, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, features),
            )
            self.t_net = copy.deepcopy(self.s_net)
            for index, layer in enumerate(self.t_net):
                if isinstance(layer, nn.Tanh):
                    self.t_net[index] = nn.ReLU()
        except Exception as e:
            raise Exception(f"ConditionalCouplingLayer - __init__ - Init error: {e}") from e

    def forward(self, x: torch.Tensor, h: torch.Tensor = None):
        try:
            x1 = x * self.mask
            
            if self.cond_size > 0 and h is not None:
                cond_input = torch.cat([h, x1], dim=-1)
            else:
                cond_input = x1

            s_raw = self.s_net(cond_input)
            scale = torch.tanh(s_raw)

            translation = self.t_net(cond_input)

            s = scale * (1 - self.mask)
            t = translation * (1 - self.mask)

            y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
            
            log_det = s.sum(dim=-1)

            return y, log_det
        except Exception as e:
            raise Exception(f"ConditionalCouplingLayer - forward - Execution error: {e}") from e

    def inverse(self, y: torch.Tensor, h: torch.Tensor = None):
        try:
            y1 = y * self.mask
            
            if self.cond_size > 0 and h is not None:
                cond_input = torch.cat([h, y1], dim=-1)
            else:
                cond_input = y1

            s_raw = self.s_net(cond_input)
            scale = torch.tanh(s_raw)

            translation = self.t_net(cond_input)

            s = scale * (1 - self.mask)
            t = translation * (1 - self.mask)

            x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
            return x
        except Exception as e:
            raise Exception(f"ConditionalCouplingLayer - inverse - Execution error: {e}") from e
