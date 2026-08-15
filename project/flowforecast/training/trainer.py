"""Training loop (paper §4.1).

Objective:
    L = 1 / (|D| * T) * sum_{x_{1:T} in D} sum_{t=1}^{T} log p_X(x_t | h_t; theta)
maximized by SGD with Adam (Kingma & Ba, 2015), end to end over the flow and the
backbone parameters.

Hyperparameters (Appendix D.2, verbatim): batch size 64, 100 batches per epoch,
max 40 epochs, learning rate 1e-3.

Gradient clipping is NOT mentioned in the paper -- it is our addition, and it
matters here because a single exploding exp(s(.)) can wreck the run.

What this trainer guarantees:
    * `lr_schedule` defaults to "onecycle", matching the reference
      implementation (zalandoresearch/pytorch-ts, pts/trainer.py), which builds
      `OneCycleLR(max_lr=10 * lr)` and steps it every batch. Adam also carries
      the reference's `weight_decay=1e-6`. This matters more than it looks: the
      lr anneals to ~0 by the final batch, so the last epoch is a stable
      stopping point BY CONSTRUCTION, and a short budget is a complete cycle
      rather than a truncated one. Our previous constant 1e-3 over 40 epochs is
      what produced runaway validation curves and measurable over-fitting
      (Solar CRPS_sum 0.69 at 40 epochs versus 0.03 for Electricity at 15).
    * `select_metric` defaults to "last". The reference keeps NO checkpoints and
      hands the FINAL weights to the predictor -- `copy_parameters(trained_net,
      prediction_net)` -- and its validation loop only prints `avg_val_loss`:
      it never compares it, never saves, never stops. There is no early stopping
      anywhere in it. So "last" is the faithful choice, and early stopping is
      disabled whenever it is active.
      The alternatives exist for ablations: "nll" picks the minimum train_nll,
      "crps" picks the minimum autoregressive validation CRPS_sum, and "both"
      writes both independent checkpoint tracks while making `_best.pt` the CRPS
      track. The legacy names "train" and "val_crps" are accepted as aliases.
      "train" picks the minimum train_nll
      (our earlier behaviour, which actively favours the most over-fit weights),
      and "val" picks the minimum val_nll. We measured why "val" is a trap:
      val_nll goes through BatchNormFlow's running statistics, which never match
      the batch statistics the weights were optimized against, so it explodes
      (Solar+MAF reached +222127 while train_nll fell monotonically to -431) and
      minimising it reliably selects epoch 1. Cost on Solar / Real-NVP:
      CRPS_sum 0.83 for the val-selected epoch-1 checkpoint versus 0.49.
    * `{run_name}_best.pt` holds the selected epoch, `{run_name}.pt` the last
      one for Colab auto-resume.
    * validation NLL is still computed and logged when a val_loader is given --
      it is a useful divergence diagnostic, it is just not a selection signal.
    * a NaN / inf loss aborts with a pointer to the usual cause instead of
      quietly filling the weights with NaN (roadmap section 7).
    * checkpoints only hold plain containers (dict / list / float / str) plus
      state dicts, so torch.load works under any torch >= 2.1 default.
"""

import copy
import math
import os
import random

import torch
import torch.nn as nn

from ..data.scaling import compute_scale, invert_scale
from ..evaluation.metrics import compute_metrics
from ..evaluation.sampling import sample_trajectories


class Trainer:
    def __init__(self, model, train_loader, val_loader=None, lr: float = 1e-3,
                 grad_clip: float = 0.0, batches_per_epoch: int = 100,
                 device: str = "cuda", early_stopping_patience: int | None = None,
                 select_metric: str = "last", weight_decay: float = 1e-6,
                 lr_schedule: str = "onecycle", max_lr: float = 1e-2,
                 validation_num_samples: int = 20,
                 validation_seed: int = 1729,
                 validation_interval: int = 1,
                 verbose: bool = True):
        try:
            self.device = torch.device(device)
            self.model = model.to(self.device)
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.lr = lr
            self.grad_clip = grad_clip
            self.batches_per_epoch = int(batches_per_epoch)
            self.early_stopping_patience = early_stopping_patience
            select_metric = str(select_metric).strip().lower()
            aliases = {"train": "nll", "val_crps": "crps"}
            select_metric = aliases.get(select_metric, select_metric)
            if select_metric not in ("last", "nll", "crps", "both"):
                raise ValueError(
                    "select_metric must be 'last', 'nll', 'crps' or 'both', "
                    f"got {select_metric!r}"
                )
            if select_metric in ("crps", "both") and val_loader is None:
                raise ValueError(
                    f"select_metric={select_metric!r} requires validation windows"
                )
            self.select_metric = select_metric
            self.lr_schedule = str(lr_schedule).strip().lower()
            if self.lr_schedule not in ("onecycle", "constant"):
                raise ValueError(
                    f"lr_schedule must be 'onecycle' or 'constant', "
                    f"got {lr_schedule!r}"
                )
            self.max_lr = float(max_lr)
            self.weight_decay = float(weight_decay)
            self.validation_num_samples = int(validation_num_samples)
            self.validation_seed = int(validation_seed)
            self.validation_interval = int(validation_interval)
            if self.validation_num_samples < 2:
                raise ValueError("validation_num_samples must be at least 2")
            if self.validation_interval < 1:
                raise ValueError("validation_interval must be at least 1")
            self.verbose = verbose
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                                              weight_decay=weight_decay)
            self.lr_scheduler = None
        except Exception as e:
            raise Exception(f"Trainer - __init__ - Init error: {e}") from e

    def fit(self, num_epochs: int = 40, resume_from: str | None = None,
            checkpoint_dir: str = "checkpoints", run_name: str = "run",
            extra_state: dict | None = None) -> dict:
        try:
            history = {"train_nll": [], "val_nll": [], "val_crps_sum": []}
            best_scores = {"nll": float("inf"), "crps": float("inf")}
            best_epochs = {"nll": -1, "crps": -1}
            start_epoch = 0
            checkpoint = None
            optimizer_state = None

            os.makedirs(checkpoint_dir, exist_ok=True)
            last_path = os.path.join(checkpoint_dir, f"{run_name}.pt")
            best_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")
            nll_path = os.path.join(checkpoint_dir, f"{run_name}_nll.pt")
            crps_path = os.path.join(checkpoint_dir, f"{run_name}_crps.pt")

            if resume_from and os.path.exists(resume_from):
                print(f"[trainer] resuming from {resume_from}")
                checkpoint = torch.load(resume_from, map_location=self.device,
                                        weights_only=False)
                saved_budget = checkpoint.get("schedule_num_epochs")
                if saved_budget is not None and int(saved_budget) != int(num_epochs):
                    raise ValueError(
                        f"checkpoint OneCycle budget is {saved_budget} epochs but "
                        f"this run requests {num_epochs}; resume requires an "
                        f"identical schedule"
                    )
                expected_contract = (extra_state or {}).get("model_contract")
                if expected_contract is not None and checkpoint.get(
                        "model_contract") != expected_contract:
                    raise ValueError(
                        f"checkpoint contract is "
                        f"{checkpoint.get('model_contract')!r}; expected "
                        f"{expected_contract!r}. Start with --no-resume."
                    )
                expected_config = (extra_state or {}).get("config")
                if expected_config is not None \
                        and checkpoint.get("config") != expected_config:
                    raise ValueError(
                        "checkpoint config differs from the requested run; "
                        "exact resume requires identical model, data, and "
                        "training configuration. Start with --no-resume."
                    )
                saved_trainer = checkpoint.get("trainer_contract") or {}
                current_trainer = self._trainer_contract(num_epochs)
                if saved_trainer and saved_trainer != current_trainer:
                    raise ValueError(
                        f"checkpoint trainer contract {saved_trainer} differs "
                        f"from requested {current_trainer}; exact resume requires "
                        f"identical settings"
                    )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer_state = checkpoint["optimizer_state_dict"]
                start_epoch = int(checkpoint.get("epoch", -1)) + 1
                loaded = checkpoint.get("history") or {}
                history = {
                    "train_nll": list(loaded.get("train_nll", [])),
                    "val_nll": list(loaded.get("val_nll", [])),
                    "val_crps_sum": list(loaded.get("val_crps_sum", [])),
                }
                saved_scores = checkpoint.get("best_scores") or {}
                saved_epochs = checkpoint.get("best_epochs") or {}
                best_scores = {
                    "nll": float(saved_scores.get(
                        "nll", checkpoint.get("best_score", float("inf"))
                    )),
                    "crps": float(saved_scores.get("crps", float("inf"))),
                }
                best_epochs = {
                    "nll": int(saved_epochs.get(
                        "nll", checkpoint.get("best_epoch", -1)
                    )),
                    "crps": int(saved_epochs.get("crps", -1)),
                }
                print(f"[trainer] resumed at epoch {start_epoch} "
                      f"(best nll {best_scores['nll']:.4f}, "
                      f"best crps {best_scores['crps']:.4f})")

            if start_epoch >= num_epochs:
                print(f"[trainer] nothing to do: checkpoint is already at epoch "
                      f"{start_epoch} of {num_epochs}")
                return history

            # One-cycle schedule over the WHOLE budget, as the reference does
            # (pts/trainer.py builds OneCycleLR with max_lr = 10x the base lr and
            # steps it every batch). Two things this buys us:
            #   * the lr anneals to ~0 by the final batch, so the last epoch is a
            #     stable stopping point by construction -- which is exactly why
            #     the reference can hand over the final weights with no selection.
            #   * the schedule is defined by `num_epochs`, so a short 12-epoch run
            #     is a COMPLETE cycle, not a truncated one. Cutting the budget to
            #     fit Colab therefore does not leave the model mid-schedule.
            # A constant 1e-3 for 40 epochs is what produced the runaway
            # validation curves and the over-fitting we measured on Solar.
            self.lr_scheduler = None
            if self.lr_schedule == "onecycle":
                self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.max_lr,
                    steps_per_epoch=self.batches_per_epoch,
                    epochs=int(num_epochs),
                )
                if checkpoint and checkpoint.get("lr_scheduler_state_dict"):
                    self.lr_scheduler.load_state_dict(
                        checkpoint["lr_scheduler_state_dict"]
                    )
                else:
                    for _ in range(start_epoch * self.batches_per_epoch):
                        self.lr_scheduler.step()
                print(f"[trainer] OneCycleLR: base_lr={self.lr:g}, "
                      f"max_lr={self.max_lr:g}, "
                       f"{num_epochs} epochs x {self.batches_per_epoch} batches")

            # OneCycleLR construction mutates optimizer LR/betas. Restore the
            # optimizer only after constructing/loading the scheduler so the
            # first resumed batch is identical to an uninterrupted run.
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)

            if checkpoint:
                self._restore_rng_state(checkpoint)

            for epoch in range(start_epoch, num_epochs):
                train_nll = self._train_one_epoch(epoch)
                history["train_nll"].append(train_nll)

                val_nll = self._validate() if self.val_loader is not None else None
                if val_nll is not None:
                    history["val_nll"].append(val_nll)

                validate_crps_now = self.val_loader is not None and (
                    (epoch + 1) % self.validation_interval == 0
                    or epoch + 1 == num_epochs
                )
                val_crps = self._validate_crps() if validate_crps_now else None
                if val_crps is not None:
                    history["val_crps_sum"].append(val_crps)

                # See the module docstring. The reference implementation keeps no
                # checkpoints at all and hands the FINAL weights to the
                # predictor, so "last" is the faithful default. "train" exists
                # because a min-train_nll pick was the previous behaviour, and
                # "val" only because it is occasionally useful for a diagnostic
                # ablation -- with BatchNorm in the flow it selects badly.
                nll_improved = train_nll < best_scores["nll"] - 1e-6
                crps_improved = val_crps is not None and math.isfinite(val_crps) \
                    and val_crps < best_scores["crps"] - 1e-8
                if nll_improved:
                    best_scores["nll"] = train_nll
                    best_epochs["nll"] = epoch
                if crps_improved:
                    best_scores["crps"] = val_crps
                    best_epochs["crps"] = epoch

                if self.select_metric == "last":
                    selected_improved = True
                elif self.select_metric in ("nll", "both"):
                    selected_improved = nll_improved
                else:
                    selected_improved = crps_improved

                if self.verbose:
                    # Put the "improved" star next to the metric that actually
                    # drives selection, and label the other one as a diagnostic.
                    # Otherwise a star printed after val_nll reads as if val_nll
                    # improved, which it very often has not.
                    nll_star = "*" if nll_improved else ""
                    crps_star = "*" if crps_improved else ""
                    line = f"train_nll: {train_nll:.4f}{nll_star}"
                    if val_crps is not None:
                        line += f" - val_crps_sum: {val_crps:.6f}{crps_star}"
                    if val_nll is not None:
                        line += f" - val_nll: {val_nll:.4f} (diagnostic)"
                    print(f"[trainer] epoch {epoch + 1}/{num_epochs} - {line}")

                state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "history": {k: [float(v) for v in vals]
                                for k, vals in history.items()},
                    "best_score": float(
                        best_scores["crps"] if self.select_metric in ("crps", "both")
                        else train_nll if self.select_metric == "last"
                        else best_scores["nll"]
                    ),
                    "best_epoch": int(
                        best_epochs["crps"] if self.select_metric in ("crps", "both")
                        else epoch if self.select_metric == "last"
                        else best_epochs["nll"]
                    ),
                    "best_scores": dict(best_scores),
                    "best_epochs": dict(best_epochs),
                    "schedule_num_epochs": int(num_epochs),
                    "trainer_contract": self._trainer_contract(num_epochs),
                    "lr_scheduler_state_dict": (
                        self.lr_scheduler.state_dict()
                        if self.lr_scheduler is not None else None
                    ),
                    "python_rng_state": random.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else None
                    ),
                }
                if extra_state:
                    state.update(copy.deepcopy(extra_state))

                self._save_checkpoint(state, last_path)
                if nll_improved:
                    self._save_checkpoint(state, nll_path)
                if crps_improved:
                    self._save_checkpoint(state, crps_path)
                if self.select_metric == "both":
                    selected_improved = crps_improved
                if selected_improved:
                    self._save_checkpoint(state, best_path)

                selected_epoch = (best_epochs["crps"]
                                  if self.select_metric in ("crps", "both")
                                  else best_epochs["nll"])
                if self.select_metric not in ("last", "both") \
                        and self.early_stopping_patience \
                        and selected_epoch >= 0 \
                        and epoch - selected_epoch >= self.early_stopping_patience:
                    print(f"[trainer] early stopping at epoch {epoch + 1}: no "
                          f"improvement for {self.early_stopping_patience} epochs "
                          f"(best epoch {selected_epoch + 1})")
                    break

            print(f"[trainer] best train NLL: {best_scores['nll']:.4f} at epoch "
                  f"{best_epochs['nll'] + 1} -> {nll_path}")
            if best_epochs["crps"] >= 0:
                print(f"[trainer] best validation CRPS_sum: "
                      f"{best_scores['crps']:.6f} at epoch "
                      f"{best_epochs['crps'] + 1} -> {crps_path}")
            print(f"[trainer] selection mode={self.select_metric} -> {best_path}")
            return history
        except Exception as e:
            raise Exception(f"Trainer - fit - Execution error: {e}") from e

    def _train_one_epoch(self, epoch: int) -> float:
        try:
            self.model.train()
            total = 0.0
            iterator = iter(self.train_loader)

            for step in range(self.batches_per_epoch):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(self.train_loader)
                    batch = next(iterator)

                x = batch["x"].to(self.device)
                c = batch["c"].to(self.device)
                observed = batch.get("observed")
                if observed is not None:
                    observed = observed.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                loss = self.model.loss(x, c, observed=observed)

                if not math.isfinite(float(loss.item())):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch + 1}, step {step + 1}. "
                        f"Almost always exp(s(.)) exploding in the coupling layer "
                        f"or BatchNorm running statistics going bad -- lower the "
                        f"learning rate or check flows/coupling.py "
                        f"(roadmap section 7)."
                    )

                loss.backward()
                gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
                if any(not torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError(
                        f"non-finite gradient at epoch {epoch + 1}, step {step + 1}"
                    )
                if self.grad_clip and self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip, error_if_nonfinite=True
                    )
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                total += float(loss.item())

            return total / self.batches_per_epoch
        except Exception as e:
            raise Exception(f"Trainer - _train_one_epoch - Execution error: {e}") from e

    def _validate(self) -> float:
        try:
            self.model.eval()
            total = 0.0
            batches = 0
            with torch.no_grad():
                for batch in self.val_loader:
                    x = batch["x"].to(self.device)
                    c = batch["c"].to(self.device)
                    observed = batch.get("observed")
                    if observed is not None:
                        observed = observed.to(self.device)
                    total += float(self.model.loss(x, c, observed=observed).item())
                    batches += 1
            return total / max(1, batches)
        except Exception as e:
            raise Exception(f"Trainer - _validate - Execution error: {e}") from e

    def _validate_crps(self) -> float:
        """Autoregressive CRPS_sum on held-out rolling validation windows."""
        try:
            dataset = self.val_loader.dataset
            required = ("starts", "series", "observed", "time_feat", "lags",
                        "context_length", "prediction_length", "max_lag")
            missing = [name for name in required if not hasattr(dataset, name)]
            if missing:
                raise ValueError(
                    f"validation dataset lacks rolling fields: {missing}"
                )

            python_state = random.getstate()
            torch_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() \
                if torch.cuda.is_available() else None
            diagnostics = os.environ.get("FLOWFORECAST_MAF_DIAGNOSTICS")
            try:
                random.seed(self.validation_seed)
                torch.manual_seed(self.validation_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.validation_seed)
                os.environ["FLOWFORECAST_MAF_DIAGNOSTICS"] = "0"

                scale_numerator = torch.zeros(
                    dataset.num_series, dtype=dataset.series.dtype
                )
                scale_denominator = torch.zeros_like(scale_numerator)
                for start in dataset.starts:
                    context = dataset.series[
                        :, start:start + dataset.context_length
                    ]
                    observed = dataset.observed[
                        :, start:start + dataset.context_length
                    ]
                    scale_numerator += (context.abs() * observed).sum(dim=-1)
                    scale_denominator += observed.sum(dim=-1)
                default_scale = scale_numerator / scale_denominator.clamp_min(1.0)

                numerator = 0.0
                denominator = 0.0
                for start in dataset.starts:
                    forecast_start = start + dataset.context_length
                    hist_start = forecast_start - dataset.context_length \
                        - dataset.max_lag
                    window_end = forecast_start + dataset.prediction_length
                    history = dataset.series[:, hist_start:forecast_start]
                    history_observed = dataset.observed[
                        :, hist_start:forecast_start
                    ]
                    target = dataset.series[:, forecast_start:window_end]
                    target_observed = dataset.observed[
                        :, forecast_start:window_end
                    ]
                    scale = compute_scale(
                        history[:, dataset.max_lag:], time_dim=-1,
                        observed=history_observed[:, dataset.max_lag:],
                        default_scale=default_scale,
                    )
                    time_feat = dataset.time_feat[
                        :, hist_start:window_end
                    ].t().unsqueeze(0)
                    samples_scaled = sample_trajectories(
                        self.model, history.t().unsqueeze(0).to(self.device),
                        time_feat.to(self.device), dataset.lags,
                        dataset.prediction_length,
                        num_samples=self.validation_num_samples,
                        context_length=dataset.context_length, scale=scale,
                    )
                    samples = invert_scale(
                        samples_scaled.cpu(), scale, time_dim=-1
                    )
                    metrics = compute_metrics(
                        samples, target, observed=target_observed
                    )
                    numerator += float(metrics["_crps_sum_numerator"])
                    denominator += float(metrics["_crps_sum_denominator"])
                return numerator / denominator if denominator > 0 else float("inf")
            finally:
                random.setstate(python_state)
                torch.set_rng_state(torch_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)
                if diagnostics is None:
                    os.environ.pop("FLOWFORECAST_MAF_DIAGNOSTICS", None)
                else:
                    os.environ["FLOWFORECAST_MAF_DIAGNOSTICS"] = diagnostics
        except Exception as e:
            raise Exception(
                f"Trainer - _validate_crps - Execution error: {e}"
            ) from e

    @staticmethod
    def _restore_rng_state(checkpoint: dict) -> None:
        try:
            if checkpoint.get("python_rng_state") is not None:
                random.setstate(checkpoint["python_rng_state"])
            if checkpoint.get("torch_rng_state") is not None:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if torch.cuda.is_available() \
                    and checkpoint.get("cuda_rng_state_all") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        except Exception as e:
            raise Exception(
                f"Trainer - _restore_rng_state - Execution error: {e}"
            ) from e

    def _trainer_contract(self, num_epochs: int) -> dict:
        return {
            "num_epochs": int(num_epochs),
            "batches_per_epoch": int(self.batches_per_epoch),
            "lr": float(self.lr),
            "max_lr": float(self.max_lr),
            "lr_schedule": self.lr_schedule,
            "weight_decay": float(self.weight_decay),
            "grad_clip": float(self.grad_clip),
            "select_metric": self.select_metric,
            "validation_num_samples": int(self.validation_num_samples),
            "validation_seed": int(self.validation_seed),
            "validation_interval": int(self.validation_interval),
        }

    @staticmethod
    def _save_checkpoint(state: dict, path: str) -> None:
        try:
            temporary = f"{path}.tmp"
            torch.save(state, temporary)
            os.replace(temporary, path)
        except Exception as e:
            raise Exception(
                f"Trainer - _save_checkpoint - Atomic save failed: {e}"
            ) from e


def load_checkpoint(model, path: str, device="cpu",
                    expected_contract: str | None = None) -> dict:
    """Load a Trainer checkpoint (or a bare state dict) into `model`."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if expected_contract is not None and (
                not isinstance(checkpoint, dict)
                or checkpoint.get("model_contract") != expected_contract):
            raise ValueError(
                f"checkpoint contract is "
                f"{checkpoint.get('model_contract') if isinstance(checkpoint, dict) else None!r}; "
                f"expected {expected_contract!r}. Retraining is required."
            )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            return checkpoint
        model.load_state_dict(checkpoint)          # bare state dict
        return {"model_state_dict": checkpoint}
    except Exception as e:
        raise Exception(f"trainer - load_checkpoint - Execution error: {e}") from e
