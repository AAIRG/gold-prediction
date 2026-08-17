"""Calendar covariates with window-independent Fourier periods.

Why this module exists instead of calling `flowforecast.data.covariates.time_features`
directly: that function derives the Fourier period from the array it is handed,

    period = max(int(values.max()) + 1, 1)

which is fine for the paper's datasets because training and evaluation always
pass the SAME full array. The gold deployment does not: the notebook builds
features over the whole Yahoo history (`dayofyear` reaches 365 or 366) while the
Streamlit app only downloads a short `3mo` window (`dayofyear` reaches maybe
230). The same calendar day therefore received a different phase at training
time and at inference time, so the app was feeding the network covariates it had
never been trained on.

Here the period of every feature is a fixed constant, so the features depend
only on the timestamp and never on how much history happens to be loaded. The
channel layout is identical to the project's function -- `cos` then `sin` per
feature, in `FEATURES` order -- so `cond_size` is unchanged and checkpoints stay
compatible with `flowforecast.utils.build.build_model`.
"""

from __future__ import annotations

import pandas as pd
import torch

# Business-daily gold data, matching FEATURES_BY_FREQ["B"] in the project.
FEATURES: tuple[str, ...] = ("dayofweek", "dayofyear")

# Fixed denominators. 7 covers Saturday and Sunday even though a business-day
# index never lands on them, and 366 covers leap years, so neither value can
# change with the length of the loaded window.
PERIODS: dict[str, int] = {
    "minute": 60,
    "hour": 24,
    "dayofweek": 7,
    "dayofmonth": 31,
    "dayofyear": 366,
}

FREQ = "B"

# Written into the exported metadata so the app can refuse to run against a
# checkpoint that was trained with a different covariate definition.
TIME_FEATURE_SPEC: dict = {
    "name": "stable_fourier_v1",
    "freq": FREQ,
    "features": list(FEATURES),
    "periods": {feature: PERIODS[feature] for feature in FEATURES},
    "channels_per_feature": 2,
    "channel_order": "cos_then_sin",
}


def time_feature_count() -> int:
    """C_time -- two Fourier channels per calendar feature."""
    try:
        return 2 * len(FEATURES)
    except Exception as e:
        raise Exception(f"features - time_feature_count - Execution error: {e}") from e


def _raw_values(dates: pd.DatetimeIndex, feature: str):
    try:
        if feature == "minute":
            return dates.minute.to_numpy()
        if feature == "hour":
            return dates.hour.to_numpy()
        if feature == "dayofweek":
            return dates.dayofweek.to_numpy()
        if feature == "dayofmonth":
            return dates.day.to_numpy()
        if feature == "dayofyear":
            return dates.dayofyear.to_numpy()
        raise ValueError(f"unknown calendar feature {feature!r}")
    except Exception as e:
        raise Exception(f"features - _raw_values - Execution error: {e}") from e


def stable_time_features(dates) -> torch.Tensor:
    """Fourier calendar features as [C_time, T], independent of window length."""
    try:
        index = dates if isinstance(dates, pd.DatetimeIndex) \
            else pd.DatetimeIndex(dates)
        if len(index) == 0:
            raise ValueError("no timestamps were supplied")
        if len(index) > 1 and (index[1] - index[0]) < pd.Timedelta("1min"):
            raise ValueError(
                "timestamps are spaced less than a minute apart -- they look "
                "like raw integers rather than real dates"
            )

        channels = []
        for feature in FEATURES:
            values = _raw_values(index, feature)
            period = float(PERIODS[feature])
            phase = torch.as_tensor(values, dtype=torch.float32) \
                * (2.0 * torch.pi / period)
            channels.extend([torch.cos(phase), torch.sin(phase)])
        return torch.stack(channels)
    except Exception as e:
        raise Exception(
            f"features - stable_time_features - Execution error: {e}"
        ) from e


def assert_spec(spec: dict | None) -> None:
    """Fail loudly when a checkpoint was trained with other covariates."""
    try:
        if spec is None:
            raise ValueError(
                "the exported metadata has no time_feature_spec, so it predates "
                "the window-independent covariates and must be retrained"
            )
        if spec != TIME_FEATURE_SPEC:
            raise ValueError(
                f"time feature spec mismatch: metadata says {spec} but this "
                f"build produces {TIME_FEATURE_SPEC}. Retraining is required."
            )
    except Exception as e:
        raise Exception(f"features - assert_spec - Execution error: {e}") from e
