"""Render the dashboard through Streamlit's AppTest harness.

This executes app.py exactly as the server would, including every tab body, and
fails if the script raises or if any Streamlit exception element is emitted. It
needs no browser and no Yahoo access: `goldflow.data.download_history` is patched
to return the synthetic series, which is also what keeps the check deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "project"))
sys.path.insert(0, str(HERE))

from run_pipeline_local import synthetic_prices  # noqa: E402


def main() -> None:
    from streamlit.testing.v1 import AppTest

    from goldflow import data

    data.download_history = lambda ticker, period="max": synthetic_prices(
        rows=400, seed=1234
    )

    harness = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    harness.run()

    if harness.exception:
        for item in harness.exception:
            print("EXCEPTION:", item.value)
        raise AssertionError("the app raised while rendering")

    titles = [element.value for element in harness.title]
    print(f"[ok] title: {titles}")

    tabs = harness.tabs
    print(f"[ok] rendered {len(tabs)} tabs")
    if len(tabs) < 7:
        raise AssertionError(f"expected at least 7 tabs, got {len(tabs)}")

    labels = [tab.label for tab in tabs]
    print(f"[ok] tab labels: {labels}")
    expected = ["LSTM-Real-NVP", "LSTM-MAF", "Transformer-MAF",
                "Vec-LSTM-ind-scaling", "Boosted ensemble", "Experiments",
                "Market data"]
    missing = [label for label in expected if label not in labels]
    if missing:
        raise AssertionError(f"missing tabs: {missing}")

    metrics = harness.metric
    print(f"[ok] {len(metrics)} metric widgets")
    if len(metrics) < 4 * len(expected[:5]):
        raise AssertionError("too few metrics: a model tab did not render")

    frames = harness.dataframe
    print(f"[ok] {len(frames)} dataframes rendered")
    if len(frames) < 10:
        raise AssertionError("too few tables: the experiments tab did not render")

    errors = harness.error
    if errors:
        for element in errors:
            print("ERROR ELEMENT:", element.value)
        raise AssertionError("the app rendered an error banner")

    warnings = [element.value for element in harness.warning]
    if warnings:
        print(f"[note] warnings shown: {warnings}")

    successes = [element.value for element in harness.success]
    print(f"[ok] success banners: {successes}")

    # The winner callout on the experiments tab must name a model.
    if not any("lowest CRPS_sum" in text for text in successes):
        raise AssertionError("the experiments tab did not report a winner")

    print("\nAPP RENDER OK")


if __name__ == "__main__":
    main()
