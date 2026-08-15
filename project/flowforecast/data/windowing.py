"""Random window sampler (paper §4.1).

"In practice, the time series x_{1:T} in a batch D are selected from a random
time window of size T within our training data, and the relative time steps are
kept constant. This allows the model to learn to cold-start given only the
covariates."

So: the window size T is fixed, the start index is random, and the model never
learns absolute position from the index -- only from covariates (§4.1).

T = context_length + prediction_length. The paper does not state
context_length; our choice is context_length = prediction_length (the GluonTS
convention). Prediction lengths per dataset are in Appendix A, Table 2.

What this module guarantees, and what it used to get wrong:

    * raw targets and lags are returned here. Batch-aware MeanScaler and target
      dequantization are applied in models/inputs.py, where the official global
      per-dimension fallback can be computed across the minibatch.
    * every window starts at index >= max(lags), and the lag channels are built
      from real history (data/covariates.py), so no lag is ever a zero pad.
    * lag channels carry raw history; models/inputs.py scales every lag block
      with the same context-derived scale used by the flow target.
    * dequantization (§5.2, count data) applies only to the density target.
    * `start_index` / `end_index` bound the sampler, which is how the caller
      keeps the test range (and the validation range) out of training.

Two datasets are provided:
    WindowedDataset        random starts, for training (§4.1)
    RollingWindowDataset   deterministic consecutive windows, for validation and
                           for the rolling-window test protocol of §5.2
"""

import random

import torch
from torch.utils.data import Dataset

from .covariates import lag_features_from_history
from .scaling import compute_scale


class _WindowBase(Dataset):
    """Shared window extraction: history -> dequantize -> scale -> covariates."""

    def __init__(self, series: torch.Tensor, time_feat: torch.Tensor,
                 context_length: int, prediction_length: int, lags: list,
                  dequantize_counts: bool = False, scaling: bool = True,
                  observed: torch.Tensor | None = None,
                 start_index: int | None = None, end_index: int | None = None):
        try:
            if series.dim() != 2:
                raise ValueError(f"series must be [D, T], got {tuple(series.shape)}")
            if time_feat.dim() != 2 or time_feat.shape[1] < series.shape[1]:
                raise ValueError(
                    "time_feat must be [C_time, T_total] covering the series"
                )
            if not lags:
                raise ValueError("lags must be a non-empty list")

            self.series = series
            if observed is None:
                observed = torch.ones_like(series)
            if observed.shape != series.shape:
                raise ValueError(
                    f"observed must match series, got {tuple(observed.shape)} "
                    f"and {tuple(series.shape)}"
                )
            self.observed = observed.to(dtype=torch.float32)
            self.time_feat = time_feat
            self.context_length = int(context_length)
            self.prediction_length = int(prediction_length)
            self.lags = list(lags)
            self.max_lag = max(self.lags)
            self.dequantize_counts = bool(dequantize_counts)
            self.scaling = bool(scaling)

            self.num_series, self.total_timesteps = series.shape
            self.window_size = self.context_length + self.prediction_length

            self.start_index = self.max_lag if start_index is None \
                else max(int(start_index), self.max_lag)
            self.end_index = self.total_timesteps if end_index is None \
                else min(int(end_index), self.total_timesteps)

            self.first_start = self.start_index
            self.last_start = self.end_index - self.window_size
            if self.last_start < self.first_start:
                raise ValueError(
                    f"usable range [{self.start_index}, {self.end_index}) is too "
                    f"short for window {self.window_size} plus max lag "
                    f"{self.max_lag}: increase the range or shorten the window"
                )
        except Exception as e:
            raise Exception(f"{type(self).__name__} - __init__ - Init error: {e}") from e

    @property
    def covariate_dim(self) -> int:
        return self.time_feat.shape[0] + self.num_series * len(self.lags)

    def make_window(self, start: int) -> dict:
        try:
            start = int(start)
            if start < self.max_lag:
                raise ValueError(
                    f"window start {start} is inside the lag warm-up region "
                    f"(max lag {self.max_lag})"
                )
            end = start + self.window_size
            if end > self.total_timesteps:
                raise ValueError(
                    f"window [{start}, {end}) exceeds series length "
                    f"{self.total_timesteps}"
                )

            raw_extended = self.series[:, start - self.max_lag:end]
            observed_extended = self.observed[:, start - self.max_lag:end]

            context = raw_extended[
                :, self.max_lag:self.max_lag + self.context_length
            ]
            if self.scaling:
                context_observed = observed_extended[
                    :, self.max_lag:self.max_lag + self.context_length
                ]
                scale = compute_scale(context, time_dim=-1,
                                       observed=context_observed)       # [D]
            else:
                scale = torch.ones(self.num_series, dtype=raw_extended.dtype)

            x = raw_extended[:, self.max_lag:]
            lag = lag_features_from_history(
                raw_extended, self.lags, self.window_size
            )
            tf = self.time_feat[:, start:end]                         # [C_time, T]
            c = torch.cat([lag, tf], dim=0)                           # [C, T]

            return {
                "x": x.t().contiguous(),        # [T, D]
                "c": c.t().contiguous(),        # [T, C]
                "observed": observed_extended[:, self.max_lag:].t().contiguous(),
                "scale": scale,                 # [D]
                "start": torch.tensor(start, dtype=torch.long),
            }
        except Exception as e:
            raise Exception(
                f"{type(self).__name__} - make_window - Execution error: {e}"
            ) from e


class WindowedDataset(_WindowBase):
    """Random windows inside [start_index, end_index) -- the §4.1 sampler."""

    def __init__(self, series: torch.Tensor, time_feat: torch.Tensor,
                 context_length: int, prediction_length: int, lags: list,
                 num_windows_per_epoch: int = 6400,
                  dequantize_counts: bool = False, scaling: bool = True,
                  observed: torch.Tensor | None = None,
                 start_index: int | None = None, end_index: int | None = None,
                 fixed_start: int | None = None):
        super().__init__(
            series, time_feat, context_length, prediction_length, lags,
            dequantize_counts=dequantize_counts, scaling=scaling,
            observed=observed, start_index=start_index, end_index=end_index,
        )
        try:
            self.num_windows_per_epoch = int(num_windows_per_epoch)
            # `fixed_start` powers --overfit-one-window: the same window every time.
            self.fixed_start = None if fixed_start is None else int(fixed_start)
        except Exception as e:
            raise Exception(f"WindowedDataset - __init__ - Init error: {e}") from e

    def __len__(self):
        return self.num_windows_per_epoch

    def __getitem__(self, idx):
        try:
            if self.fixed_start is not None:
                return self.make_window(self.fixed_start)
            return self.make_window(random.randint(self.first_start, self.last_start))
        except Exception as e:
            raise Exception(f"WindowedDataset - __getitem__ - Execution error: {e}") from e


class RollingWindowDataset(_WindowBase):
    """Deterministic consecutive windows -- validation and the §5.2 test protocol.

    Windows are laid out backwards from `end_index` with a stride of
    `prediction_length` (default), so the last window ends exactly at
    `end_index` and the windows tile the evaluation range without overlap of the
    prediction parts.
    """

    def __init__(self, series: torch.Tensor, time_feat: torch.Tensor,
                 context_length: int, prediction_length: int, lags: list,
                 num_windows: int = 5, stride: int | None = None,
                 dequantize_counts: bool = False, scaling: bool = True,
                 observed: torch.Tensor | None = None,
                 start_index: int | None = None, end_index: int | None = None):
        super().__init__(
            series, time_feat, context_length, prediction_length, lags,
            dequantize_counts=dequantize_counts, scaling=scaling,
            observed=observed, start_index=start_index, end_index=end_index,
        )
        try:
            self.stride = int(stride) if stride else self.prediction_length
            starts = []
            for i in range(int(num_windows)):
                start = self.last_start - (int(num_windows) - 1 - i) * self.stride
                if start >= self.first_start:
                    starts.append(start)
            if not starts:
                raise ValueError("no valid rolling window fits in the given range")
            self.starts = starts
        except Exception as e:
            raise Exception(f"RollingWindowDataset - __init__ - Init error: {e}") from e

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        try:
            return self.make_window(self.starts[int(idx)])
        except Exception as e:
            raise Exception(
                f"RollingWindowDataset - __getitem__ - Execution error: {e}"
            ) from e
