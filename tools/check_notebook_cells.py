"""Execute the notebook's own cell sources locally, on CPU, with a tiny budget.

Reads `train_gold_yahoo.ipynb`, skips only the Colab-specific cells (Drive mount
and `files.download`), swaps the training budget for a small one, and runs the
remaining cells in order in a shared namespace. That covers the data and candle
assertions, the paper tables, the leaderboard, the boosting report, the figures,
and the horizon-reconstruction proof -- the parts most likely to break in Colab.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "project"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from goldflow import pipeline, variants  # noqa: E402
from run_pipeline_local import synthetic_prices  # noqa: E402

NOTEBOOK = ROOT / "train_gold_yahoo.ipynb"
SKIP_MARKERS = ("drive.mount", "files.download")

SMALL = dict(
    seeds=(1, 2), epochs=2, batch_size=32, batches_per_epoch=6,
    test_windows=6, validation_windows=12, evaluation_samples=40,
    validation_samples=10, ensemble_rounds=8, boost_fit_windows=40,
    boost_gate_windows=15, boost_samples=16, device="cpu",
)


def clean(source: str) -> str:
    return "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line
        for line in source.splitlines()
    )


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="goldflow_nb_"))
    displayed: list = []

    namespace = {
        "__name__": "__notebook__",
        "np": np, "pd": pd, "torch": torch, "shutil": shutil, "sys": sys,
        "Path": Path, "json": json,
        "pipeline": pipeline, "variants": variants,
        "display": lambda *args: displayed.extend(args),
        "WORK_ROOT": ROOT,
    }

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    executed = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        title = source.splitlines()[0].replace("#@title ", "").strip()
        if any(marker in source for marker in SKIP_MARKERS):
            print(f"[skip] cell {index}: {title}  (Colab only)")
            continue

        print(f"[run ] cell {index}: {title}")
        exec(compile(clean(source), f"cell{index}", "exec"), namespace)
        executed += 1

        if "PipelineConfig(" in source:
            config = namespace["CONFIG"]
            namespace["CONFIG"] = dataclasses.replace(
                config, work_dir=work / "run", export_dir=work / "artifacts",
                prices=synthetic_prices(rows=1400, seed=11), **SMALL,
            )
            print(f"       -> budget replaced with the local test settings "
                  f"({SMALL['epochs']} epochs, seeds {SMALL['seeds']})")

    print(f"\n[ok] executed {executed} notebook cells")
    print(f"[ok] {len(displayed)} display() payloads produced")
    if len(displayed) < 8:
        raise AssertionError("too few tables were displayed")

    report = namespace["REPORT"]
    for name in ("table1_crps_sum", "table4_crps", "table5_mse"):
        if name not in report["tables"]:
            raise AssertionError(f"{name} was not built")
    print(f"[ok] paper tables: {sorted(report['tables'])}")

    figures = sorted(p.name for p in (work / "artifacts" / "figures").glob("*.png"))
    expected = {"prediction_intervals.png", "learning_curves.png",
                "cross_covariance_boosted.png"}
    missing = expected - set(figures)
    if missing:
        raise AssertionError(f"missing figures: {sorted(missing)}")
    print(f"[ok] figures written: {figures}")

    live = namespace["live"]
    if set(live) != set(variants.variant_names()):
        raise AssertionError(f"the live forecast covers only {sorted(live)}")
    print(f"[ok] live forecasts for {sorted(live)}")

    board = report["leaderboard"]
    print("\n[leaderboard]")
    print(board[["Rank", "Model", "crps_sum_mean", "runs"]].to_string(index=False))
    print("\n" + pipeline.format_winner(report))

    shutil.rmtree(work, ignore_errors=True)
    print("\nNOTEBOOK CELLS OK")


if __name__ == "__main__":
    main()
