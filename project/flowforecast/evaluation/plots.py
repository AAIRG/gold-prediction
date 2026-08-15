"""Plots for the report and the presentation.

Three figures mirror the paper:
    - Fan chart of prediction intervals, like Figures 8-11 in the appendix
      (observations vs prediction with quantile bands).
    - Covariance / cross-covariance heatmaps for the toy pipes experiment, like
      Figure 4 (ground truth next to fitted model, darker = higher positive
      value).
    - Cross-covariance of a real dataset vs the mean of 100 sampled trajectories,
      like Figure 5 (data | samples | absolute difference).

Shape contract (the old version silently mixed these up and crashed inside
matplotlib):
    history  [D, T_ctx]  or [T_ctx]     ground truth before the forecast start
    target   [D, P]      or [P]         ground truth over the horizon
    samples  [S, D, P]                  sampled trajectories, original units
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _series(array, series_idx: int):
    """Pick one series out of [D, T] / [T] and return 1-D numpy."""
    array = _to_numpy(array)
    if array is None:
        return None
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array[series_idx]
    raise ValueError(f"expected [D, T] or [T], got shape {array.shape}")


def plot_fan_chart(history, target, samples, series_idx: int = 0,
                   save_path: str | None = None, title: str | None = None,
                   quantiles=(0.1, 0.5, 0.9)):
    try:
        samples_np = _to_numpy(samples)
        if samples_np.ndim != 3:
            raise ValueError(
                f"samples must be [S, D, P], got shape {samples_np.shape}"
            )
        low, mid, high = quantiles
        q_low = np.quantile(samples_np[:, series_idx, :], low, axis=0)
        q_mid = np.quantile(samples_np[:, series_idx, :], mid, axis=0)
        q_high = np.quantile(samples_np[:, series_idx, :], high, axis=0)

        hist = _series(history, series_idx)
        tgt = _series(target, series_idx)
        n_hist = 0 if hist is None else len(hist)
        x_pred = np.arange(n_hist, n_hist + len(q_mid))

        plt.figure(figsize=(10, 4))
        if hist is not None:
            plt.plot(np.arange(n_hist), hist, color="black", label="History")
        plt.fill_between(x_pred, q_low, q_high, color="tab:blue", alpha=0.25,
                         label=f"{int(low * 100)}-{int(high * 100)}%")
        plt.plot(x_pred, q_mid, color="tab:blue", label="Median")
        if tgt is not None:
            plt.plot(x_pred[:len(tgt)], tgt, color="tab:red", label="Observed")
        if n_hist:
            plt.axvline(n_hist - 0.5, color="grey", linewidth=0.8, linestyle="--")

        plt.title(title or f"Forecast, series {series_idx}")
        plt.xlabel("time step")
        plt.legend()
        plt.tight_layout()
        _finish(save_path)
    except Exception as e:
        raise Exception(f"plots - plot_fan_chart - Execution error: {e}") from e


def plot_loss_curve(history: dict, save_path: str | None = None,
                    title: str = "Learning curve"):
    try:
        plt.figure(figsize=(8, 5))
        train = history.get("train_nll") if hasattr(history, "get") else None
        if train:
            plt.plot(range(1, len(train) + 1), train, label="Train NLL")
        val = history.get("val_nll") if hasattr(history, "get") else None
        if val:
            plt.plot(range(1, len(val) + 1), val, label="Validation NLL")
        plt.xlabel("epoch")
        plt.ylabel("negative log-likelihood")
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        _finish(save_path)
    except Exception as e:
        raise Exception(f"plots - plot_loss_curve - Execution error: {e}") from e


def plot_covariance_comparison(ground_truth, fitted, labels=None,
                               save_path: str | None = None,
                               show_diff: bool = True,
                               titles=("Data", "Samples"),
                               suptitle: str | None = None):
    """Figure 4 / Figure 5: heatmaps on a shared scale plus |difference|."""
    try:
        gt = _to_numpy(ground_truth)
        fit = _to_numpy(fitted)
        if gt.shape != fit.shape:
            raise ValueError(
                f"matrices must have the same shape, got {gt.shape} and {fit.shape}"
            )

        n_panels = 3 if show_diff else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.6))
        vmin = float(min(gt.min(), fit.min()))
        vmax = float(max(gt.max(), fit.max()))

        for ax, matrix, panel_title in zip(axes[:2], (gt, fit), titles):
            im = ax.imshow(matrix, cmap="Blues", vmin=vmin, vmax=vmax)
            ax.set_title(panel_title)
            _ticks(ax, labels, matrix.shape[0])
            fig.colorbar(im, ax=ax, fraction=0.046)

        if show_diff:
            diff = np.abs(gt - fit)
            im = axes[2].imshow(diff, cmap="Reds", vmin=0.0, vmax=float(diff.max()))
            axes[2].set_title("|difference|")
            _ticks(axes[2], labels, diff.shape[0])
            fig.colorbar(im, ax=axes[2], fraction=0.046)

        if suptitle:
            fig.suptitle(suptitle)
        fig.tight_layout()
        _finish(save_path)
    except Exception as e:
        raise Exception(
            f"plots - plot_covariance_comparison - Execution error: {e}"
        ) from e


def plot_series_grid(series, num_series: int = 6, save_path: str | None = None,
                     title: str = "Simulated pipes (§5.1)"):
    """Quick look at a few series -- used by the toy pipes experiment (Figure 3)."""
    try:
        data = _to_numpy(series)
        num_series = min(num_series, data.shape[0])
        fig, axes = plt.subplots(num_series, 1, figsize=(9, 1.5 * num_series),
                                 sharex=True)
        if num_series == 1:
            axes = [axes]
        for i in range(num_series):
            axes[i].plot(data[i], linewidth=0.9)
            axes[i].set_ylabel(f"s{i}")
        axes[-1].set_xlabel("time step")
        fig.suptitle(title)
        fig.tight_layout()
        _finish(save_path)
    except Exception as e:
        raise Exception(f"plots - plot_series_grid - Execution error: {e}") from e


def _ticks(ax, labels, size: int) -> None:
    if not labels:
        return
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)


def _finish(save_path: str | None) -> None:
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[plots] wrote {save_path}")
    else:
        plt.show()
