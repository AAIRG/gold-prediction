"""Run one account's fair reduced-budget Colab assignment."""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time


VARIANTS = {
    "lstm_realnvp": ("lstm", "realnvp"),
    "lstm_maf": ("lstm", "maf"),
    "transformer_maf": ("transformer", "maf"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", required=True)
    parser.add_argument("--budgets-json", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--drive-project", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--variants", nargs="+", default=["transformer_maf"],
                        choices=["lstm_realnvp", "lstm_maf", "transformer_maf"])
    parser.add_argument("--run-baseline", action="store_true")
    return parser.parse_args()


def run(command: list[str], sync_callback=None, label: str | None = None) -> None:
    printable = " ".join(command)
    task = label or printable
    started = time.monotonic()
    print(f"\n[runner] START {task}", flush=True)
    print(f">>> {printable}", flush=True)
    stop = threading.Event()

    def periodic_sync():
        elapsed_seconds = 0
        while not stop.wait(60):
            elapsed_seconds += 60
            minutes, seconds = divmod(elapsed_seconds, 60)
            print(
                f"[runner] HEARTBEAT {task} | elapsed={minutes:02d}:{seconds:02d}",
                flush=True,
            )
            if sync_callback and elapsed_seconds % 300 == 0:
                sync_callback()

    worker = threading.Thread(target=periodic_sync, daemon=True)
    worker.start()
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(command, check=True, env=env)
    finally:
        stop.set()
        worker.join(timeout=1)
        if sync_callback:
            sync_callback()
        elapsed = time.monotonic() - started
        print(f"[runner] END {task} | elapsed={elapsed / 60:.1f} min", flush=True)


def make_config(dataset: str, variant: str, backbone: str, flow: str,
                 epochs: int, account: str, batch_size: int,
                 batches_per_epoch: int) -> str:
    text = pathlib.Path(f"configs/{dataset}.yaml").read_text(encoding="utf-8")
    text = re.sub(r"(?m)^  type: (realnvp|maf).*$", f"  type: {flow}", text)
    text = re.sub(
        r"(?m)^  type: (lstm|transformer).*$", f"  type: {backbone}", text
    )
    text = re.sub(r"(?m)^  epochs: .*$", f"  epochs: {epochs}", text)
    text = re.sub(
        r"(?m)^  batch_size: .*$", f"  batch_size: {batch_size}", text
    )
    text = re.sub(
        r"(?m)^  batches_per_epoch: .*$",
        f"  batches_per_epoch: {batches_per_epoch}", text,
    )
    if account == "account1":
        selection = {
            "val_windows": "5",
            "select_metric": "both",
            "validation_num_samples": "100",
            "validation_seed": "1729",
            "validation_interval": "5",
        }
    else:
        selection = {
            "val_windows": "0",
            "select_metric": "last",
            "validation_num_samples": "20",
            "validation_seed": "1729",
            "validation_interval": "1",
        }
    for key, value in selection.items():
        pattern = rf"(?m)^  {re.escape(key)}: .*$"
        replacement = f"  {key}: {value}"
        if re.search(pattern, text):
            text = re.sub(pattern, replacement, text)
        else:
            training_match = re.search(r"(?m)^training:\s*$", text)
            if training_match:
                insert_at = training_match.end()
                text = (text[:insert_at] + "\n" + replacement
                        + text[insert_at:])
            else:
                text += f"\ntraining:\n{replacement}\n"
    path = pathlib.Path(f"configs/{dataset}__{variant}__{account}.yaml")
    path.write_text(text, encoding="utf-8")
    return str(path)


def sync_to_drive(args) -> None:
    try:
        for path in (args.outputs, args.checkpoints):
            if os.path.isdir(path):
                destination = os.path.join(args.drive_project, path)
                shutil.copytree(path, destination, dirs_exist_ok=True)
        print(f"[sync] {args.account} artifacts copied to Drive", flush=True)
    except Exception as e:
        raise Exception(
            f"run_colab_account - sync_to_drive - Copy failed: {e}"
        ) from e


def run_dataset(args, dataset: str, budget: dict) -> dict[str, str]:
    epochs = int(budget["epochs"])
    seeds = [int(seed) for seed in budget["seeds"]]
    print(
        f"\n[runner] DATASET {dataset} | epochs={epochs} | seeds={seeds} | "
        f"batch_size={args.batch_size} | "
        f"batches_per_epoch={args.batches_per_epoch}",
        flush=True,
    )
    selected = {variant: VARIANTS[variant] for variant in args.variants}
    configs = {
        variant: make_config(
            dataset, variant, backbone, flow, epochs, args.account,
            args.batch_size, args.batches_per_epoch,
        )
        for variant, (backbone, flow) in selected.items()
    }
    seed_args = [str(seed) for seed in seeds]
    suffix = f"e{epochs}_b{args.batches_per_epoch}_bs{args.batch_size}"
    common = [
        "--seeds", *seed_args,
        "--run-suffix", suffix,
        "--outputs", args.outputs,
        "--checkpoints", args.checkpoints,
        "--skip-aggregate",
    ]
    failures = {}
    for variant, config in configs.items():
        try:
            run([sys.executable, "scripts/run_experiments.py", "--config", config,
                 *common], sync_callback=lambda: sync_to_drive(args),
                label=f"{dataset} / {variant}")
        except Exception as error:
            failures[variant] = str(error)
            print(f"[FAILED] {dataset} / {variant}: {error}", flush=True)
    if args.run_baseline and "lstm_realnvp" in configs:
        try:
            run([
                sys.executable, "scripts/run_experiments.py",
                "--config", configs["lstm_realnvp"],
                "--baseline", "veclstm", *common,
            ], sync_callback=lambda: sync_to_drive(args),
                label=f"{dataset} / veclstm")
        except Exception as error:
            failures["veclstm"] = str(error)
            print(f"[FAILED] {dataset} / veclstm: {error}", flush=True)
    print(
        f"[runner] DATASET COMPLETE {dataset} | failures={sorted(failures)}",
        flush=True,
    )
    return failures


def main():
    args = parse_args()
    try:
        budgets = json.loads(args.budgets_json)
        print(
            f"[runner] ACCOUNT {args.account} | datasets={list(budgets)} | "
            "live_output=enabled | heartbeat=60s | drive_sync=300s",
            flush=True,
        )
        failures = {}
        for dataset, budget in budgets.items():
            try:
                dataset_failures = run_dataset(args, dataset, budget)
                if dataset_failures:
                    failures[dataset] = dataset_failures
            except Exception as error:
                failures[dataset] = {"orchestration": str(error)}
                print(f"[FAILED] {dataset}: {error}", flush=True)
        output_path = os.path.join(args.outputs, "account_status.json")
        os.makedirs(args.outputs, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump({"account": args.account, "failures": failures}, handle,
                      indent=2, sort_keys=True)
        sync_to_drive(args)
        if failures:
            raise RuntimeError(
                f"one or more dataset runs did not complete: {failures}"
            )
    except Exception as e:
        raise Exception(
            f"run_colab_account - main - Account sweep failed: {e}"
        ) from e


if __name__ == "__main__":
    main()
