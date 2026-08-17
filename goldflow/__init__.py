"""Gold-specific layer on top of the unchanged `project/flowforecast` code.

`project/` is the verbatim paper implementation and is never edited. Everything
that is specific to the gold deployment lives here so that the Colab notebook
and the Streamlit app share exactly one definition of

    * the multi-horizon return target,
    * the calendar covariates,
    * the model variants,
    * the price-path reconstruction and the derived horizon returns,
    * the rolling-window experiment protocol and the paper tables,
    * the boosting layers.

Duplicating any of those between the notebook and the app is what previously
let training and inference drift apart, so nothing here is re-implemented on
either side of the fence.
"""

from .boosting import (BoostingResult, GradientResidualBooster,
                       fit_greedy_crps_ensemble, mix_samples)
from .data import (RETURN_HORIZONS, aggregate_candles, build_return_targets,
                   download_history, latest_close)
from .experiments import (SeedSummary, build_paper_tables, coverage_report,
                          evaluate_samples, leaderboard, paper_table,
                          per_dimension_metrics, rolling_window_samples,
                          summarize_windows)
from .features import (TIME_FEATURE_SPEC, stable_time_features,
                       time_feature_count)
from .forecast import (ForecastResult, consistency_report,
                       derive_horizon_returns, ensemble_forecast,
                       forecast_prices, probability_frame, quantile_frame)
from .variants import (MODEL_VARIANTS, base_config, build_variant,
                       variant_config, variant_names)

__all__ = [
    "BoostingResult",
    "GradientResidualBooster",
    "fit_greedy_crps_ensemble",
    "mix_samples",
    "RETURN_HORIZONS",
    "aggregate_candles",
    "build_return_targets",
    "download_history",
    "latest_close",
    "SeedSummary",
    "build_paper_tables",
    "coverage_report",
    "evaluate_samples",
    "leaderboard",
    "paper_table",
    "per_dimension_metrics",
    "rolling_window_samples",
    "summarize_windows",
    "TIME_FEATURE_SPEC",
    "stable_time_features",
    "time_feature_count",
    "ForecastResult",
    "consistency_report",
    "derive_horizon_returns",
    "ensemble_forecast",
    "forecast_prices",
    "probability_frame",
    "quantile_frame",
    "MODEL_VARIANTS",
    "base_config",
    "build_variant",
    "variant_config",
    "variant_names",
]
