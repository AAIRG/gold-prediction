"""Run every verification step in order, end to end, on CPU.

Builds a throwaway artifact export with a tiny training budget, points the app at
it through `GOLDFLOW_ARTIFACTS`, then runs each check as its own subprocess so a
failure names the step that broke. Nothing is written into `artifacts/`, so the
repository never ends up shipping models trained on the synthetic series.

    python tools/verify_all.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

STEPS = [
    ("unit checks: horizon maths, candles, covariates, four variants",
     ["tools/smoke_test.py"], False),
    ("notebook: every cell compiles and resolves against goldflow",
     ["tools/check_notebook.py"], False),
    ("pipeline: full Colab run at a tiny budget",
     ["tools/run_pipeline_local.py", "--keep"], True),
    ("notebook: execute the real cell sources",
     ["tools/check_notebook_cells.py"], False),
    ("app: loading, forecasting, reconstruction, tables",
     ["tools/check_app.py"], False),
    ("app: render every tab through Streamlit AppTest",
     ["tools/check_app_render.py"], False),
]


def main() -> None:
    artifacts = Path(tempfile.mkdtemp(prefix="goldflow_verify_"))
    environment = dict(os.environ)
    environment["GOLDFLOW_ARTIFACTS"] = str(artifacts)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(HERE)])
    environment["MPLBACKEND"] = "Agg"

    results = []
    for label, command, produces_artifacts in STEPS:
        arguments = list(command)
        if produces_artifacts:
            arguments += ["--export-into", str(artifacts)]
        print(f"\n{'=' * 78}\n== {label}\n{'=' * 78}")
        started = time.time()
        completed = subprocess.run(
            [sys.executable, *arguments], cwd=str(ROOT), env=environment,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        elapsed = time.time() - started
        tail = [
            line for line in (completed.stdout or "").splitlines()
            if line.startswith(("[ok]", "[skip]", "[export]"))
            or line.endswith(("OK", "PASSED"))
        ]
        print("\n".join(tail[-14:]) or (completed.stdout or "")[-1500:])
        if completed.returncode != 0:
            print("\n--- stdout tail ---")
            print((completed.stdout or "")[-3000:])
            print("--- stderr tail ---")
            print((completed.stderr or "")[-3000:])
            raise SystemExit(f"FAILED: {label}")
        results.append((label, elapsed))
        print(f"-> passed in {elapsed:.1f}s")

    print(f"\n{'=' * 78}")
    for label, elapsed in results:
        print(f"  PASS  {elapsed:6.1f}s  {label}")
    print(f"{'=' * 78}\nALL VERIFICATION STEPS PASSED")

    shutil.rmtree(artifacts, ignore_errors=True)


if __name__ == "__main__":
    main()
