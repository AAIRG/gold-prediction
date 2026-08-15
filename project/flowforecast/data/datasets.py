"""Dataset loaders: raw source -> [D, total_timesteps] tensor + real start date.

Properties per Appendix A, Table 2 of the paper:

    dataset      D      domain   freq      total steps   prediction length
    Exchange     8      R+       daily     6,071         30
    Solar        137    R+       hourly    7,009         24
    Electricity  370    R+       hourly    5,790         24
    Traffic      963    (0,1)    hourly    10,413        24
    Taxi         1,214  N        30-min    1,488         24
    Wikipedia    2,000  N        daily     792           30

CAREFUL WITH THAT "total steps" COLUMN: it is the length of the TRAINING range,
not of the whole array. Verified against the data: `exchange_rate_nips` has train
entries of exactly 6,071 steps, and its last rolling test entry has
6,071 + 5 * 30 = 6,221. So the array this module returns is longer than the table
by `rolling_evaluations * prediction_length`, and cutting exactly that many steps
off the end (`train_test_split` with num_test_windows = rolling_evaluations)
reproduces the published split.

§5.2: data is "preprocessed exactly as in Salinas et al. (2019a)". That sentence
decides where the files come from: the numbers above are the GP-Copula
(mv_release) versions shipped by GluonTS as `*_nips`, NOT the raw UCI / LSTNet
files. The LSTNet electricity file has 321 series and the LSTNet traffic file has
862, so loading those and asserting D = 370 / 963 can only ever fail.

    exchange  -> exchange_rate_nips  (8,    freq B,     5 rolling evaluations)
    solar     -> solar_nips          (137,  freq h,     7)
    electric  -> electricity_nips    (370,  freq h,     7)
    traffic   -> traffic_nips        (963,  freq h,     7)
    taxi      -> taxi_30min          (1214, freq 30min, 56)
    wikipedia -> wiki2000_nips       (2000, freq D,     5)

The GluonTS `*_nips` files ship as D UNIVARIATE series, and the test split is a
rolling evaluation. Grouping them into one multivariate array is done by
GluonTS's own `MultivariateGrouper`, which is exactly what the GP-Copula code of
Salinas et al. (2019a) uses -- and §5.2 says the data is "preprocessed exactly as
in Salinas et al. (2019a)", so this is the faithful path rather than a taste.

Do NOT try to reconstruct the multivariate array by hand-slicing `ds.test`. We
did, and it was wrong: the raw test entries are ordered SERIES-major, not
date-major (`MultivariateGrouper._prepare_test_data` recovers one date with the
stride `dataset[test_start::num_test_dates]`). Keeping "the last D entries"
therefore mixes different series with different rolling dates, and the
`min(len(...))` truncation that follows silently shortens the array -- Electricity
came out as 370 x 1249 instead of 370 x 5958, with a reported training range
(5521) longer than the whole array, which is self-contradictory.

`grouper_test` returns one multivariate entry per rolling date, each one
prediction_length longer than the last, and it left-pads so the result is
rectangular. We keep the LAST entry -- the full series including every test
window -- and split it ourselves with `train_test_split`, so training never
touches the test range.

Count data (Taxi, Wikipedia) is flagged with `is_count`; §5.2 dequantizes it
before mean-scaling (see data/scaling.py, applied in data/windowing.py).

Fallbacks: Exchange and Solar can also be built from the public LSTNet files
(Lai et al., 2018) when gluonts is not installed. For Electricity / Traffic /
Taxi / Wikipedia there is no faithful public fallback, so those raise with an
actionable message instead of silently loading a differently-shaped file.

Acceptance criterion (M0): shape matches the table and there are no NaNs.
"""

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch


class DatasetSpec(NamedTuple):
    """One row of Appendix A, Table 2 plus where to get the data.

    `total_timesteps` is Table 2's column, i.e. the length of the TRAINING range.
    The array the loaders return has `full_length` steps: the training range plus
    the `rolling_evaluations` test blocks.

    The first four fields are ordered so that legacy tuple unpacking
    `D, freq, prediction_length, total_timesteps = DATASET_SPECS[name]` keeps
    working.
    """
    num_series: int
    freq: str
    prediction_length: int
    total_timesteps: int
    gluonts_name: str | None = None
    rolling_evaluations: int = 1
    is_count: bool = False
    fallback_url: str | None = None
    fallback_start: str | None = None
    fallback_aggregate: int = 1

    @property
    def full_length(self) -> int:
        return self.total_timesteps + self.rolling_evaluations * self.prediction_length


DATASET_SPECS: dict[str, DatasetSpec] = {
    "exchange": DatasetSpec(
        num_series=8, freq="B", prediction_length=30, total_timesteps=6071,
        gluonts_name="exchange_rate_nips", rolling_evaluations=5,
        fallback_url="https://raw.githubusercontent.com/laiguokun/"
                     "multivariate-time-series-data/master/exchange_rate/"
                     "exchange_rate.txt.gz",
        fallback_start="1990-01-01",
    ),
    "solar": DatasetSpec(
        num_series=137, freq="H", prediction_length=24, total_timesteps=7009,
        gluonts_name="solar_nips", rolling_evaluations=7,
        fallback_url="https://raw.githubusercontent.com/laiguokun/"
                     "multivariate-time-series-data/master/solar-energy/"
                     "solar_AL.txt.gz",
        fallback_start="2006-01-01", fallback_aggregate=6,  # 10-min -> hourly
    ),
    "electricity": DatasetSpec(
        num_series=370, freq="H", prediction_length=24, total_timesteps=5790,
        gluonts_name="electricity_nips", rolling_evaluations=7,
    ),
    "traffic": DatasetSpec(
        num_series=963, freq="H", prediction_length=24, total_timesteps=10413,
        gluonts_name="traffic_nips", rolling_evaluations=7,
    ),
    "taxi": DatasetSpec(
        num_series=1214, freq="30min", prediction_length=24, total_timesteps=1488,
        gluonts_name="taxi_30min", rolling_evaluations=56, is_count=True,
    ),
    "wikipedia": DatasetSpec(
        num_series=2000, freq="D", prediction_length=30, total_timesteps=792,
        gluonts_name="wiki2000_nips", rolling_evaluations=5, is_count=True,
    ),
}


@dataclass
class MultivariateDataset:
    """Everything downstream code needs about a loaded dataset."""
    name: str
    values: torch.Tensor        # [D, full_length] = training range + test blocks
    start: pd.Timestamp         # real calendar start (Appendix D.1 needs it)
    freq: str                   # canonical frequency key
    prediction_length: int
    is_count: bool = False
    rolling_evaluations: int = 1
    train_length: int | None = None   # Table 2's "total steps", when known

    # The grouped TRAIN split, kept separately. GluonTS groups train and test
    # with two MultivariateGrouper instances, each of which computes its own
    # `first_timestamp`, so the two arrays live on DIFFERENT calendar origins and
    # their lengths are not comparable. The reference implementation never merges
    # them: it trains on the train array and evaluates each grouped test entry on
    # its own. Keeping both here is what makes that possible -- and it is the
    # only way to express Taxi's published 56-window protocol, where the test
    # array (1469 steps) is far too short to also hold the training range.
    # `None` for the raw-file fallback, which really is one contiguous array.
    train_values: torch.Tensor | None = None
    train_start: pd.Timestamp | None = None
    observed_values: torch.Tensor | None = None
    train_observed_values: torch.Tensor | None = None

    @property
    def num_series(self) -> int:
        return int(self.values.shape[0])

    @property
    def total_timesteps(self) -> int:
        return int(self.values.shape[1])

    @property
    def train_timesteps(self) -> int | None:
        """Length of the grouped train array, when it is available."""
        if self.train_values is None:
            return None
        return int(self.train_values.shape[1])

    def summary(self) -> str:
        extra = ""
        if self.train_values is not None:
            extra = (f", train_array={self.train_timesteps}"
                     f"@{None if self.train_start is None else self.train_start.date()}")
        return (f"{self.name}: D={self.num_series}, T={self.total_timesteps} "
                f"(training range {self.train_length}), freq={self.freq}, "
                f"start={self.start.date()}, "
                f"prediction_length={self.prediction_length}, "
                f"rolling_evaluations={self.rolling_evaluations}, "
                f"count_data={self.is_count}{extra}")


def load_dataset(name: str, cache_dir: str | None = None,
                 fallback_path: str | None = None,
                 prefer_gluonts: bool = True,
                 strict: bool = False,
                 require_gluonts: bool = False) -> MultivariateDataset:
    """Load one of the six datasets of Table 2.

    `strict=True` turns the Table-2 shape mismatch into an error; the default
    only warns, so a slightly different local copy still runs end to end.
    """
    try:
        key = str(name).strip().lower()
        if key not in DATASET_SPECS:
            raise ValueError(
                f"Unknown dataset {name!r}. Known: {sorted(DATASET_SPECS)}"
            )
        spec = DATASET_SPECS[key]

        bundle = None
        gluonts_error = None
        if prefer_gluonts and spec.gluonts_name:
            try:
                bundle = _load_from_gluonts(key, spec, cache_dir)
            except Exception as e:                       # noqa: BLE001
                gluonts_error = e

        if bundle is None:
            if require_gluonts:
                raise RuntimeError(
                    f"the reproduction contract requires GluonTS dataset "
                    f"{spec.gluonts_name!r}; loading failed: {gluonts_error}"
                )
            if not spec.fallback_url:
                raise RuntimeError(
                    f"{key} can only be loaded through gluonts "
                    f"('{spec.gluonts_name}'): the public raw files have a "
                    f"different number of series than Table 2. Install gluonts "
                    f"(pip install gluonts) and retry. "
                    f"Underlying error: {gluonts_error}"
                )
            print(f"[datasets] gluonts path unavailable ({gluonts_error}); "
                  f"falling back to the public raw file for {key}.")
            bundle = _load_from_raw_file(key, spec, fallback_path)

        _verify(bundle, spec, strict=strict)
        return bundle
    except Exception as e:
        raise Exception(f"datasets - load_dataset - Execution error: {e}") from e


def _load_from_gluonts(key: str, spec: DatasetSpec,
                       cache_dir: str | None) -> MultivariateDataset:
    """The full rolling-evaluation series of a `*_nips` dataset -> [D, T].

    Grouping is done by GluonTS's own MultivariateGrouper; see the module
    docstring for why hand-slicing `ds.test` is wrong.
    """
    try:
        from gluonts.dataset.multivariate_grouper import MultivariateGrouper
    except ImportError as e:                               # pragma: no cover
        raise RuntimeError(
            "MultivariateGrouper is required to group the *_nips files into one "
            "multivariate array. Install a recent gluonts "
            "(pip install 'gluonts>=0.13'). Underlying import error: "
            f"{e}"
        ) from e
    from gluonts.dataset.repository.datasets import get_dataset

    from .covariates import normalize_freq

    kwargs = {"regenerate": False}
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        kwargs["path"] = Path(cache_dir)
    ds = get_dataset(spec.gluonts_name, **kwargs)

    freq = normalize_freq(getattr(ds.metadata, "freq", spec.freq) or spec.freq)
    prediction_length = int(getattr(ds.metadata, "prediction_length", None)
                            or spec.prediction_length)

    num_dates = _num_test_dates(ds, spec)

    # One multivariate entry per rolling date. The grouper left-pads, so every
    # entry is rectangular [D, T_i] with T_i growing by prediction_length.
    train_grouped = list(
        MultivariateGrouper(max_target_dim=spec.num_series)(ds.train))
    test_grouped = list(
        MultivariateGrouper(num_test_dates=num_dates,
                            max_target_dim=spec.num_series)(ds.test))
    if not test_grouped:
        raise RuntimeError(f"gluonts test split of {spec.gluonts_name} is empty")

    # The last rolling date carries the whole series, every test window included.
    last = test_grouped[-1]
    values = np.asarray(last["target"], dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise RuntimeError(
            f"{spec.gluonts_name}: grouped target has shape {values.shape}, "
            f"expected [D, T]"
        )
    observed_values = np.isfinite(values).astype(np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    start = _to_timestamp(last["start"])

    # Table 2's "total steps" is the TRAINING range. Keep the whole grouped train
    # array, not just its length: it lives on its own calendar origin and is what
    # training should actually consume (see MultivariateDataset.train_values).
    train_values = None
    train_start = None
    train_length = None
    train_observed_values = None
    if train_grouped:
        train_target = np.asarray(train_grouped[0]["target"], dtype=np.float64)
        if train_target.ndim == 1:
            train_target = train_target.reshape(1, -1)
        train_observed = np.isfinite(train_target).astype(np.float32)
        train_target = np.nan_to_num(train_target, nan=0.0, posinf=0.0, neginf=0.0)
        train_values = torch.tensor(train_target, dtype=torch.float32)
        train_observed_values = torch.tensor(train_observed, dtype=torch.float32)
        train_start = _to_timestamp(train_grouped[0]["start"])
        train_length = int(train_target.shape[-1])
    if train_length is None:
        train_length = int(values.shape[1] - num_dates * prediction_length)

    print(f"[datasets] loaded {spec.gluonts_name} from gluonts: "
          f"{values.shape[0]} series x {values.shape[1]} steps "
          f"(training range {train_length}, {num_dates} rolling test dates)")

    return MultivariateDataset(
        name=key,
        values=torch.tensor(values, dtype=torch.float32),
        start=start,
        freq=freq,
        prediction_length=prediction_length,
        is_count=spec.is_count,
        rolling_evaluations=num_dates,
        train_length=train_length,
        train_values=train_values,
        train_start=train_start,
        observed_values=torch.tensor(observed_values, dtype=torch.float32),
        train_observed_values=train_observed_values,
    )


def _first_train_length(ds) -> int | None:
    try:
        for entry in ds.train:
            return int(np.asarray(entry["target"]).reshape(-1).shape[0])
    except Exception:                                       # noqa: BLE001
        return None
    return None


def _num_train_entries(ds) -> int:
    try:
        return len(ds.train)
    except TypeError:
        return sum(1 for _ in ds.train)


def _num_test_dates(ds, spec: DatasetSpec) -> int:
    try:
        n_test = len(ds.test)
        n_train = _num_train_entries(ds)
        if n_train:
            return max(1, n_test // n_train)
    except TypeError:
        pass
    return max(1, spec.rolling_evaluations)


def _to_timestamp(start) -> pd.Timestamp:
    if isinstance(start, pd.Period):
        return start.to_timestamp()
    return pd.Timestamp(start)


def _offset(freq: str):
    from .covariates import pandas_freq
    return pd.tseries.frequencies.to_offset(pandas_freq(freq))


def _load_from_raw_file(key: str, spec: DatasetSpec,
                        fallback_path: str | None) -> MultivariateDataset:
    """LSTNet-style text file (rows = time, cols = series) -> Table 2 layout."""
    from .covariates import normalize_freq

    path = fallback_path or f"data_raw/{key}.txt.gz"
    if not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        print(f"[datasets] downloading {spec.fallback_url} -> {path}")
        urllib.request.urlretrieve(spec.fallback_url, path)

    frame = None
    for compression in ("gzip", "infer"):
        try:
            frame = pd.read_csv(path, header=None, compression=compression)
            break
        except Exception:                                  # noqa: BLE001
            continue
    if frame is None:
        raise RuntimeError(f"could not parse {path} as a headerless CSV")

    values = frame.to_numpy(dtype=np.float64).T            # [D, T]

    if spec.fallback_aggregate > 1:
        # Solar: the public file is 10-minutely, Table 2 is hourly.
        step = spec.fallback_aggregate
        usable = (values.shape[1] // step) * step
        values = values[:, :usable].reshape(values.shape[0], -1, step).sum(axis=2)

    # Keep the training range plus the rolling test blocks, to mirror the gluonts
    # split (Table 2's "total steps" is the training length only).
    if values.shape[1] > spec.full_length:
        values = values[:, :spec.full_length]
    if values.shape[0] > spec.num_series:
        values = values[:spec.num_series]

    observed_values = np.isfinite(values).astype(np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    return MultivariateDataset(
        name=key,
        values=torch.tensor(values, dtype=torch.float32),
        start=pd.Timestamp(spec.fallback_start or "2000-01-01"),
        freq=normalize_freq(spec.freq),
        prediction_length=spec.prediction_length,
        is_count=spec.is_count,
        rolling_evaluations=spec.rolling_evaluations,
        train_length=min(spec.total_timesteps,
                         values.shape[1] - spec.rolling_evaluations
                         * spec.prediction_length),
        observed_values=torch.tensor(observed_values, dtype=torch.float32),
    )


def _verify(bundle: MultivariateDataset, spec: DatasetSpec,
            strict: bool = False) -> None:
    problems = []
    if bundle.num_series != spec.num_series:
        problems.append(f"D={bundle.num_series} (Table 2: {spec.num_series})")
    # Table 2's number is the training range; the array carries the test blocks
    # too, so compare the training length, not the full array length.
    if bundle.train_length is not None \
            and bundle.train_length != spec.total_timesteps:
        problems.append(
            f"training range={bundle.train_length} "
            f"(Table 2: {spec.total_timesteps})"
        )
    if torch.isnan(bundle.values).any():
        problems.append("NaNs in the target")
    if not problems:
        return
    message = f"{bundle.name}: " + ", ".join(problems)
    if strict:
        raise AssertionError(message)
    print(f"[datasets] WARNING {message}. Continuing with the loaded shape; "
          f"pass strict=True to make this fatal.")


def load_exchange(path: str | None = None, **kwargs) -> MultivariateDataset:
    try:
        return load_dataset("exchange", fallback_path=path, **kwargs)
    except Exception as e:
        raise Exception(f"datasets - load_exchange - Execution error: {e}") from e


def load_solar(path: str | None = None, **kwargs) -> MultivariateDataset:
    try:
        return load_dataset("solar", fallback_path=path, **kwargs)
    except Exception as e:
        raise Exception(f"datasets - load_solar - Execution error: {e}") from e


def load_electricity(path: str | None = None, **kwargs) -> MultivariateDataset:
    try:
        return load_dataset("electricity", fallback_path=path, **kwargs)
    except Exception as e:
        raise Exception(f"datasets - load_electricity - Execution error: {e}") from e


def load_traffic(path: str | None = None, **kwargs) -> MultivariateDataset:
    try:
        return load_dataset("traffic", fallback_path=path, **kwargs)
    except Exception as e:
        raise Exception(f"datasets - load_traffic - Execution error: {e}") from e


def load_taxi(path: str | None = None, **kwargs) -> MultivariateDataset:
    try:
        return load_dataset("taxi", fallback_path=path, **kwargs)
    except Exception as e:
        raise Exception(f"datasets - load_taxi - Execution error: {e}") from e


def load_wikipedia(path: str | None = None, **kwargs) -> MultivariateDataset:
    try:
        return load_dataset("wikipedia", fallback_path=path, **kwargs)
    except Exception as e:
        raise Exception(f"datasets - load_wikipedia - Execution error: {e}") from e


def train_test_split(series: torch.Tensor, prediction_length: int,
                     num_test_windows: int):
    """§5.2: rolling-window prediction starting from the last training point.

    Returns (train, test) where `train` ends exactly where the first test window
    starts, so `train.shape[1]` is the forecast start index of window 0.
    """
    try:
        total = int(series.shape[1])
        test_length = int(prediction_length) * int(num_test_windows)
        if test_length >= total:
            raise ValueError(
                f"test range {test_length} does not fit in {total} timesteps"
            )
        test_start = total - test_length
        return series[:, :test_start], series[:, test_start:]
    except Exception as e:
        raise Exception(f"datasets - train_test_split - Execution error: {e}") from e
