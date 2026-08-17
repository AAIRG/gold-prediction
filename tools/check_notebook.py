"""Static validation of train_gold_yahoo.ipynb.

Compiles every code cell so a syntax error cannot reach Colab, then resolves the
`goldflow` and `flowforecast` attributes the notebook touches against the real
modules. Colab-only names are declared rather than imported, so the check runs on
any machine.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "project"))

NOTEBOOK = ROOT / "train_gold_yahoo.ipynb"

# Provided by Colab or by an earlier cell at runtime.
RUNTIME_NAMES = {
    "google", "drive", "files", "display", "get_ipython",
}


def code_cells(notebook: dict) -> list:
    cells = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        # Strip IPython magics, which are not valid Python.
        lines = [
            "" if line.lstrip().startswith(("%", "!")) else line
            for line in source.splitlines()
        ]
        cells.append((index, source, "\n".join(lines)))
    return cells


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = code_cells(notebook)
    markdown = sum(1 for c in notebook["cells"] if c["cell_type"] == "markdown")
    print(f"[ok] {NOTEBOOK.name}: {len(cells)} code cells, {markdown} markdown cells")

    combined = []
    for index, _, cleaned in cells:
        try:
            compile(cleaned, f"cell{index}", "exec")
        except SyntaxError as error:
            raise AssertionError(
                f"cell {index} has a syntax error at line {error.lineno}: "
                f"{error.msg}\n  {error.text}"
            ) from error
        combined.append(cleaned)
    print("[ok] every code cell compiles")

    tree = ast.parse("\n".join(combined))

    # Names the notebook binds itself, plus builtins and runtime injections.
    bound = set(dir(__builtins__)) | RUNTIME_NAMES
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.comprehension) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, ast.Tuple):
            for element in node.elts:
                if isinstance(element, ast.Name) and isinstance(element.ctx, ast.Store):
                    bound.add(element.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            for group in (arguments.posonlyargs, arguments.args,
                          arguments.kwonlyargs):
                bound.update(argument.arg for argument in group)
            for special in (arguments.vararg, arguments.kwarg):
                if special is not None:
                    bound.add(special.arg)

    undefined = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in bound and not node.id.startswith("_"):
                undefined.add(node.id)
    if undefined:
        raise AssertionError(f"the notebook reads undefined names: {sorted(undefined)}")
    print("[ok] no undefined names")

    # Resolve every `module.attribute` the notebook uses against the real modules.
    import flowforecast.data.toy_pipes  # noqa: F401
    import flowforecast.evaluation.plots  # noqa: F401

    from goldflow import data, forecast, pipeline, variants

    modules = {
        "pipeline": pipeline, "variants": variants, "data": data,
        "forecast": forecast,
    }
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id not in modules:
            continue
        module = modules[node.value.id]
        if not hasattr(module, node.attr):
            raise AssertionError(
                f"{node.value.id}.{node.attr} does not exist "
                f"(line {node.lineno} of the combined source)"
            )
        checked += 1
    print(f"[ok] resolved {checked} goldflow attribute accesses")

    # PipelineConfig must accept exactly the keywords the config cell passes.
    config_cell = next(source for _, source, _ in cells if "PipelineConfig(" in source)
    call = next(
        node for node in ast.walk(ast.parse(config_cell.replace("Path(", "str(")))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "PipelineConfig"
    )
    passed = {keyword.arg for keyword in call.keywords}
    fields = set(pipeline.PipelineConfig.__dataclass_fields__)
    unknown = passed - fields
    if unknown:
        raise AssertionError(f"PipelineConfig has no fields {sorted(unknown)}")
    print(f"[ok] PipelineConfig accepts all {len(passed)} configured keywords")

    # Report keys the notebook indexes must be produced by pipeline.run.
    used_keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id == "REPORT" \
                and isinstance(node.slice, ast.Constant):
            used_keys.add(node.slice.value)
    produced = {
        "labels", "model_order", "tables", "leaderboard", "coverage",
        "per_window", "per_seed", "per_horizon", "learning_curves", "summaries",
        "winner", "bundle", "export", "models",
    }
    missing = used_keys - produced
    if missing:
        raise AssertionError(f"pipeline.run does not produce REPORT keys {sorted(missing)}")
    print(f"[ok] REPORT keys used by the notebook: {sorted(used_keys)}")

    print("\nNOTEBOOK OK")


if __name__ == "__main__":
    main()
