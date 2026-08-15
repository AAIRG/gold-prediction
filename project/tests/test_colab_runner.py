"""Static orchestration tests for the reduced-budget Colab runners."""

from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts import run_colab_account
from scripts.run_colab_account import make_config


def test_make_config_applies_one_shared_budget(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "solar.yaml").write_text(
        "dataset:\n  name: solar\nflow:\n  type: realnvp\n"
        "  maf_log_scale_bound: 5.0\n"
        "backbone:\n  type: lstm\n  hidden_size: 40\n  d_model: 32\n"
        "training:\n  epochs: 40\n"
        "  batch_size: 64\n  batches_per_epoch: 100\n"
        "  grad_clip: 0.0\n  maf_grad_clip: 10.0\n"
            "  lr_schedule: onecycle\n  val_windows: 3\n"
            "  select_metric: both\n  validation_num_samples: 20\n"
        "  validation_seed: 1729\n"
        "  max_lr: 1.0e-2\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    generated = make_config(
        "solar", "transformer_maf", "transformer", "maf", 15,
        "account1", 32, 75,
    )
    config = yaml.safe_load(Path(generated).read_text(encoding="utf-8"))

    assert config["flow"]["type"] == "maf"
    assert config["backbone"]["type"] == "transformer"
    assert config["backbone"]["hidden_size"] == 40
    assert config["backbone"]["d_model"] == 32
    assert config["flow"]["maf_log_scale_bound"] == 5.0
    assert config["training"]["epochs"] == 15
    assert config["training"]["batch_size"] == 32
    assert config["training"]["batches_per_epoch"] == 75
    assert config["training"]["grad_clip"] == 0.0
    assert config["training"]["maf_grad_clip"] == 10.0
    assert config["training"]["max_lr"] == 1.0e-2
    assert config["training"]["val_windows"] == 5
    assert config["training"]["select_metric"] == "both"
    assert config["training"]["validation_num_samples"] == 100
    assert config["training"]["validation_interval"] == 5


def test_non_account1_config_keeps_official_last_selection(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "solar.yaml").write_text(
        "dataset:\n  name: solar\nflow:\n  type: realnvp\n"
        "backbone:\n  type: lstm\ntraining:\n  epochs: 40\n"
        "  select_metric: both\n  val_windows: 3\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    generated = make_config(
        "solar", "lstm_maf", "lstm", "maf", 12,
        "account2", 32, 75,
    )
    config = yaml.safe_load(Path(generated).read_text(encoding="utf-8"))

    assert config["training"]["epochs"] == 12
    assert config["training"]["select_metric"] == "last"
    assert config["training"]["val_windows"] == 0
    assert config["training"]["validation_interval"] == 1


def test_run_streams_unbuffered_child_output(monkeypatch, capsys):
    captured = {}

    def fake_subprocess_run(command, check, env):
        captured["command"] = command
        captured["check"] = check
        captured["env"] = env

    monkeypatch.setattr(
        run_colab_account.subprocess, "run", fake_subprocess_run
    )

    run_colab_account.run(["python", "child.py"], label="diagnostic task")

    assert captured["command"] == ["python", "child.py"]
    assert captured["check"] is True
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    output = capsys.readouterr().out
    assert "[runner] START diagnostic task" in output
    assert "[runner] END diagnostic task" in output


def test_dataset_runner_continues_after_one_variant_fails(monkeypatch):
    calls = []

    monkeypatch.setattr(
        run_colab_account, "make_config",
        lambda dataset, variant, *_args: f"{dataset}-{variant}.yaml",
    )
    monkeypatch.setattr(run_colab_account, "sync_to_drive", lambda _args: None)

    def fake_run(command, sync_callback=None, label=None):
        del command, sync_callback
        calls.append(label)
        if label == "solar / lstm_maf":
            raise RuntimeError("intentional failure")

    monkeypatch.setattr(run_colab_account, "run", fake_run)
    args = SimpleNamespace(
        account="account1", batch_size=64, batches_per_epoch=100,
        outputs="outputs", checkpoints="checkpoints",
        drive_project="drive",
    )

    failures = run_colab_account.run_dataset(
        args, "solar", {"epochs": 30, "seeds": [1]}
    )

    assert calls == [
        "solar / lstm_realnvp", "solar / lstm_maf",
        "solar / transformer_maf", "solar / veclstm",
    ]
    assert list(failures) == ["lstm_maf"]
