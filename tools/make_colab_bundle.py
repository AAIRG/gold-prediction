"""Zip `project/` and `goldflow/` into the single archive Colab needs.

The notebook wants one upload, not two, and it must get the model implementation
and the gold layer at exactly the revision the app runs. Caches, checkpoints and
trained artifacts are excluded so the bundle stays small and can never carry a
stale model into a fresh training run.

    python tools/make_colab_bundle.py

Writes `gold_yahoo_colab.zip` next to this folder, then upload it to
`/content/drive/MyDrive/fixed_notebooks/gold_yahoo_colab.zip`.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

INCLUDED = ("project", "goldflow")
SKIP_DIRECTORIES = {
    "__pycache__", ".git", ".pytest_cache", "checkpoints", "outputs",
    "data_raw", ".ipynb_checkpoints",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".pt", ".joblib", ".zip"}


def should_skip(path: Path) -> bool:
    try:
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            return True
        return path.suffix.lower() in SKIP_SUFFIXES
    except Exception as e:
        raise Exception(f"make_colab_bundle - should_skip - Execution error: {e}") from e


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(ROOT / "gold_yahoo_colab.zip"))
    args = parser.parse_args()

    try:
        output = Path(args.output)
        missing = [name for name in INCLUDED if not (ROOT / name).is_dir()]
        if missing:
            raise FileNotFoundError(f"missing source directories: {missing}")

        written = 0
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in INCLUDED:
                base = ROOT / name
                for path in sorted(base.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(ROOT)
                    if should_skip(relative):
                        continue
                    archive.write(path, str(relative).replace("\\", "/"))
                    written += 1

        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"[bundle] wrote {output} with {written} files ({size_mb:.2f} MB)")
        print("[bundle] upload it to "
              "/content/drive/MyDrive/fixed_notebooks/gold_yahoo_colab.zip")
    except Exception as e:
        raise Exception(f"make_colab_bundle - main - Execution error: {e}") from e


if __name__ == "__main__":
    main()
