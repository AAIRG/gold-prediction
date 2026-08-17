"""Run the real Colab pipeline locally, on CPU, with a tiny budget.

Same `goldflow.pipeline.run` the notebook calls, only with two epochs and a
handful of windows, so the whole orchestration -- training four variants across
seeds, rolling-window scoring, both boosting stages, the paper tables and the
artifact export -- is exercised without a GPU or a Colab session.

The synthetic option keeps it offline; pass --live to pull real Yahoo data.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "project"))

from goldflow import data, pipeline  # noqa: E402


def synthetic_prices(rows: int = 1400, seed: int = 11) -> pd.DataFrame:
    """A random walk with mild volatility clustering, in gold's price range."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=rows)
    volatility = 0.006 * np.exp(0.35 * np.cumsum(rng.normal(0, 0.05, rows)))
    volatility = np.clip(volatility, 0.003, 0.02)
    shocks = rng.normal(0.0002, 1.0, rows) * volatility
    close = 1600.0 * np.exp(np.cumsum(shocks))
    wick = np.abs(rng.normal(0.004, 0.0015, rows)) * close
    return pd.DataFrame(
        {
            "Open": close * np.exp(rng.normal(0, 0.0018, rows)),
            "High": close + wick,
            "Low": close - wick,
            "Close": close,
        },
        index=dates,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="download real Yahoo data instead of the synthetic walk")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--keep", action="store_true",
                        help="keep the exported artifacts instead of deleting them")
    parser.add_argument("--into-artifacts", action="store_true",
                        help="export straight into gold-yahoo/artifacts for the app")
    parser.add_argument("--export-into", type=str, default=None,
                        help="export into this directory instead of a temporary one")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work = Path(tempfile.mkdtemp(prefix="goldflow_pipeline_"))
    if args.export_into:
        export = Path(args.export_into)
    elif args.into_artifacts:
        export = ROOT / "artifacts"
    else:
        export = work / "artifacts"

    prices = None if args.live else synthetic_prices()
    config = pipeline.PipelineConfig(
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=32,
        batches_per_epoch=6,
        test_windows=6,
        validation_windows=12,
        evaluation_samples=40,
        validation_samples=10,
        ensemble_rounds=8,
        boost_fit_windows=40,
        boost_gate_windows=15,
        boost_samples=16,
        download_period="10y",
        device="cpu",
        work_dir=work,
        export_dir=export,
        prices=prices,
    )

    report = pipeline.run(config)

    print("\n" + "=" * 78)
    for name, payload in report["tables"].items():
        print(f"\n[{name}] {payload['caption']}")
        print(payload["frame"].to_string())

    print("\n[leaderboard]")
    board = report["leaderboard"]
    print(board[["Rank", "Model", "crps_sum_mean", "crps_sum_stderr",
                 "crps_mean", "mse_mean", "runs"]].to_string(index=False))

    print("\n[calibration]")
    print(report["coverage"].to_string(index=False))

    print("\n" + pipeline.format_winner(report))

    exported = sorted(p.name for p in (export / "experiments").glob("*"))
    print(f"\n[export] {export}")
    print(f"[export] models: {sorted(p.name for p in (export / 'models').glob('*'))}")
    print(f"[export] results: {exported}")

    metadata = (export / "gold_yahoo_metadata.json")
    if not metadata.exists():
        raise AssertionError("metadata was not exported")

    required = {"table1_crps_sum.csv", "table4_crps.csv", "table5_mse.csv",
                "leaderboard.csv", "coverage.csv", "per_window_metrics.csv",
                "per_seed.csv", "per_horizon.csv", "learning_curves.json"}
    missing = required - set(exported)
    if missing:
        raise AssertionError(f"missing exported results: {sorted(missing)}")

    keys = set(board["model_key"])
    if not {"lstm_realnvp", "lstm_maf", "transformer_maf", "veclstm"} <= keys:
        raise AssertionError(f"the leaderboard is missing variants: {keys}")
    if "boosted" not in keys:
        raise AssertionError("the boosted ensemble is missing from the leaderboard")

    print("\nPIPELINE OK")
    if args.keep or args.into_artifacts or args.export_into:
        print(f"[export] kept working directory {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
