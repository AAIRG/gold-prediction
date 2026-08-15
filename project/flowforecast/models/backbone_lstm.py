"""TempFlow-compatible LSTM backbone.

    h_t = RNN(concat(lags_t / scale, target_dimension_embedding, time_feat_t), h_{t-1})

The paper allows LSTM or GRU. Hyperparameters are deferred to Salinas et al.
(2019a) -- Appendix D.2 only says "The LSTM hyperparameters were the ones from
Salinas et al. (2019a)". So hidden_size / num_layers are OUR choice
(2 layers, hidden 40 is the common value there); say so in the presentation.

    input_size = D (number of series)
    cond_size  = covariate dim = C_time + D * len(lags)  (data/covariates.py)

forward() is teacher forced over the whole window and returns h for every t,
because the §4.1 loss sums over t = 1..T, i.e. context AND prediction range.
step() is the single-step version used by the §4.3 sampling loop; both share the
same alignment: feeding (x_t, c_t) produces h_{t+1}.
"""

import torch
import torch.nn as nn

class LSTMBackbone(nn.Module):
    loss_uses_observed_mask = True

    def __init__(self, input_size: int, cond_size: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.0,
                 time_feature_size: int = 0):
        super().__init__()
        try:
            self.input_size = input_size
            self.cond_size = cond_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.time_feature_size = int(time_feature_size)

            self.target_embedding = nn.Embedding(input_size, 1)
            self.rnn_input_size = cond_size + input_size
            self.lstm = nn.LSTM(input_size=self.rnn_input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout if num_layers > 1 else 0.0,
                                batch_first=True)
        except Exception as e:
            raise Exception(f"LSTMBackbone - __init__ - Init error: {e}") from e

    def forward(self, x: torch.Tensor, covariates: torch.Tensor,
                context_length: int | None = None):
        """[B, T, D] + aligned c_t [B, T, C] -> h [B, T, hidden_size]."""
        try:
            lstm_in = self._inputs(covariates)
            h, _ = self.lstm(lstm_in)
            return h
        except Exception as e:
            raise Exception(f"LSTMBackbone - forward - Execution error: {e}") from e

    def step(self, x_t: torch.Tensor, c_t: torch.Tensor, state=None):
        """One step of the rollout using conditioning for the target time.

        x_t is [B, 1, D], c_t is [B, 1, C]. The returned h is the conditioning
        for time t + 1, which is exactly the alignment forward() produces.
        """
        try:
            del x_t
            lstm_in = self._inputs(c_t)
            h_next, new_state = self.lstm(lstm_in, state)
            return h_next, new_state
        except Exception as e:
            raise Exception(f"LSTMBackbone - step - Execution error: {e}") from e

    def _inputs(self, covariates: torch.Tensor) -> torch.Tensor:
        try:
            if covariates.shape[-1] != self.cond_size:
                raise ValueError(
                    f"expected {self.cond_size} aligned covariates, got "
                    f"{covariates.shape[-1]}"
                )
            batch = covariates.shape[0]
            emb = self.target_embedding.weight[:, 0].view(1, 1, -1)
            emb = emb.expand(batch, covariates.shape[1], -1)
            split = covariates.shape[-1] - self.time_feature_size
            lag_part = covariates[:, :, :split]
            time_part = covariates[:, :, split:]
            return torch.cat([lag_part, emb, time_part], dim=-1)
        except Exception as e:
            raise Exception(f"LSTMBackbone - _inputs - Execution error: {e}") from e
