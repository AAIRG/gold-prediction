"""Run the full protocol over several seeds, with optional aggregation.

    python scripts/run_experiments.py --config configs/exchange.yaml --seeds 1 2 3
    python scripts/run_experiments.py --config configs/exchange.yaml --seeds 1 2 3 \
        --baseline veclstm --epochs 10

The paper reports "mean and standard error over 20 reruns" for CRPS_sum (Table 1)
and 3 reruns for CRPS (Table 4). A standard deviation over the 5 rolling test
windows of ONE run is a different quantity and cannot be compared with those
error bars, so this script trains and evaluates one model per seed and aggregates
across runs:

    mean          = average of the per-run CRPS_sum
    standard error = sample std / sqrt(number of runs)

Every run always writes metrics_<run>.json and an upserted results.csv row.
Unless --skip-aggregate is passed, the script also writes an aggregate JSON and
Markdown table.  Per-account Colab notebooks skip that optional report step and
defer mean/SE calculation until all accounts have been merged.
"""

import argparse
import os
import statistics
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flowforecast.utils.build import model_tag, run_name, save_json
from flowforecast.utils.common import load_config

SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--baseline", type=str, default=None, choices=["veclstm"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--outputs", type=str, default="outputs")
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    parser.add_argument(
        "--run-suffix", type=str, default=None,
        help="append a budget/version suffix to every run and checkpoint name",
    )
    parser.add_argument("--skip-training", action="store_true",
                         help="only re-evaluate existing checkpoints")
    parser.add_argument(
        "--skip-aggregate", action="store_true",
        help=("keep per-run metrics/results.csv only; defer mean/SE reporting "
              "until outputs from all Colab accounts are merged"),
    )
    return parser.parse_args()


def run(command: list) -> None:
    printable = " ".join(command)
    print(f"\n[run_experiments] $ {printable}\n")
    result = subprocess.run(command, cwd=os.path.dirname(SCRIPTS) or ".")
    if result.returncode != 0:
        raise RuntimeError(f"command failed with code {result.returncode}: {printable}")


def main():
    args = parse_args()
    try:
        config = load_config(args.config)
        dataset_name = str((config.get("dataset") or {}).get("name", "dataset"))
        tag = model_tag(config, args.baseline)
        results = []

        for seed in args.seeds:
            name = run_name(
                config, baseline=args.baseline, seed=seed,
                suffix=args.run_suffix,
            )

            if not args.skip_training:
                train_cmd = [sys.executable, os.path.join("scripts", "train.py"),
                             "--config", args.config, "--seed", str(seed),
                             "--run-name", name,
                             "--checkpoints", args.checkpoints,
                             "--outputs", args.outputs]
                if args.baseline:
                    train_cmd += ["--baseline", args.baseline]
                if args.epochs:
                    train_cmd += ["--epochs", str(args.epochs)]
                if args.device:
                    train_cmd += ["--device", args.device]
                run(train_cmd)

            selection_mode = str(
                (config.get("training") or {}).get("select_metric", "last")
            ).lower()
            if selection_mode == "both":
                checkpoints = [
                    ("last", os.path.join(args.checkpoints, f"{name}.pt")),
                    ("nll", os.path.join(args.checkpoints, f"{name}_nll.pt")),
                    ("crps", os.path.join(args.checkpoints, f"{name}_crps.pt")),
                ]
            else:
                checkpoint = os.path.join(args.checkpoints, f"{name}_best.pt")
                if not os.path.exists(checkpoint):
                    checkpoint = os.path.join(args.checkpoints, f"{name}.pt")
                checkpoints = [(selection_mode, checkpoint)]

            for selection, checkpoint in checkpoints:
                if not os.path.exists(checkpoint):
                    raise RuntimeError(
                        f"{selection}-selected checkpoint was not created: "
                        f"{checkpoint}"
                    )
                eval_name = (f"{name}__selected_{selection}"
                             if selection_mode == "both" else name)
                eval_cmd = [
                    sys.executable, os.path.join("scripts", "evaluate.py"),
                    "--config", args.config, "--checkpoint", checkpoint,
                    "--run-name", eval_name, "--seed", str(seed),
                    "--outputs", args.outputs,
                ]
                if args.baseline:
                    eval_cmd += ["--baseline", args.baseline]
                if args.num_samples:
                    eval_cmd += ["--num-samples", str(args.num_samples)]
                if args.device:
                    eval_cmd += ["--device", args.device]
                run(eval_cmd)

                payload = _read_metrics(os.path.join(
                    args.outputs, f"metrics_{eval_name}.json"
                ))
                results.append({
                    "seed": seed, "run": eval_name, "selection": selection,
                    "checkpoint": checkpoint,
                    "crps_sum": payload["summary"]["crps_sum"]["mean"],
                    "crps": payload["summary"]["crps"]["mean"],
                    "mse": payload["summary"]["mse"]["mean"],
                })

        aggregate = {"dataset": dataset_name, "model": tag,
                     "model_contract": "tempflow_v5_official_parity",
                     "metric_contract": "gluonts_multivariate_19_quantiles",
                     "num_runs": len(results), "runs": results}
        selections = sorted({r["selection"] for r in results})
        aggregate["selections"] = {}
        for selection in selections:
            selected_runs = [r for r in results if r["selection"] == selection]
            summary = {"num_runs": len(selected_runs)}
            for key in ("crps_sum", "crps", "mse"):
                values = [r[key] for r in selected_runs]
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                summary[key] = {
                    "mean": statistics.fmean(values),
                    "std": std,
                    "stderr": std / (len(values) ** 0.5),
                }
            aggregate["selections"][selection] = summary

        primary_selection = "last" if "last" in selections else selections[0]
        aggregate["primary_selection"] = primary_selection
        for key in ("crps_sum", "crps", "mse"):
            aggregate[key] = aggregate["selections"][primary_selection][key]

        if not args.skip_aggregate:
            save_json(aggregate, os.path.join(
                args.outputs, f"aggregate_{dataset_name}_{tag}.json"))
            _write_markdown(
                os.path.join(args.outputs, "results_table.md"), aggregate
            )

        print(f"\n[run_experiments] {dataset_name} / {tag}:")
        for selection in selections:
            summary = aggregate["selections"][selection]
            print(f"  [{selection}-selected] {summary['num_runs']} run(s)")
            print(f"    CRPS_sum {summary['crps_sum']['mean']:.4f} "
                  f"+/- {summary['crps_sum']['stderr']:.4f}")
            print(f"    CRPS     {summary['crps']['mean']:.4f} "
                  f"+/- {summary['crps']['stderr']:.4f}")
            print(f"    MSE      {summary['mse']['mean']:.4g} "
                  f"+/- {summary['mse']['stderr']:.4g}")
    except Exception as e:
        raise Exception(f"run_experiments - main - Sweep failed: {e}") from e


def _read_metrics(path: str) -> dict:
    import json
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_markdown(path: str, aggregate: dict) -> None:
    header = ("| dataset | model | runs | CRPS_sum (mean +/- SE) | "
              "CRPS (mean +/- SE) | MSE |\n"
              "| --- | --- | --- | --- | --- | --- |\n")
    row = (f"| {aggregate['dataset']} | {aggregate['model']} | "
           f"{aggregate['num_runs']} | "
           f"{aggregate['crps_sum']['mean']:.4f} +/- "
           f"{aggregate['crps_sum']['stderr']:.4f} | "
           f"{aggregate['crps']['mean']:.4f} +/- "
           f"{aggregate['crps']['stderr']:.4f} | "
           f"{aggregate['mse']['mean']:.4g} |\n")
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            rows = [line for line in handle if line.startswith("|")]
        rows = [line for line in rows
                if line.startswith("| dataset") or line.startswith("| ---")
                or not line.startswith(
                    f"| {aggregate['dataset']} | {aggregate['model']} |"
                )]
    if not rows:
        rows = header.splitlines(keepends=True)
    rows.append(row)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Results (mean +/- standard error over reruns)\n\n")
        handle.writelines(rows)
    print(f"[run_experiments] upserted a row in {path}")


if __name__ == "__main__":
    main()
