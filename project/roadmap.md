# Roadmap -- Conditioned Normalizing Flows for multivariate time series

Reproduction of *Multivariate Probabilistic Time Series Forecasting via
Conditioned Normalizing Flows* (Rasul, Sheikh, Schuster, Bergmann & Vollgraf,
ICLR 2021, arXiv:2002.06103v3), from scratch in PyTorch. No `gluonts` /
`pytorch-ts` model code: gluonts is used only as the source of the datasets the
paper uses, and the reference repo only for cross-checking behaviour.

Docstrings across the package point back to this file ("roadmap M4",
"roadmap section 7"); the section numbers below are those references.

---

## 1. Goal

    p_X(x_{t0:T} | x_{1:t0-1}, c_{1:T}; theta) = prod_{t=t0}^{T} p_X(x_t | h_t; theta)   (eq. 4)

`h_t` comes from an LSTM (eq. 3) or a causally masked Transformer; `p_X(x_t|h_t)`
is a conditional Real NVP or MAF acting on the D-dimensional cross-section. All
temporal structure lives in the backbone, all cross-series structure in the flow.
Variants reported by the paper: **LSTM-Real-NVP**, **LSTM-MAF**,
**Transformer-MAF**; baseline for us: **Vec-LSTM-ind-scaling** (a diagonal
Gaussian head on the same backbone).

Training objective (§4.1), maximized with Adam end to end:

    L = 1 / (|D| * T) * sum_{x_{1:T} in D} sum_{t=1}^{T} log p_X(x_t | h_t; theta)

The sum starts at t = 1: the context window is scored too, not only the horizon.

---

## 2. Layout

    flowforecast/
      flows/       coupling.py, batchnorm.py, made.py, maf.py, flow.py     (§2.1)
      models/      inputs.py, backbone_lstm.py, backbone_transformer.py,
                   conditioned_flow_model.py, vec_lstm_model.py             (§4)
      data/        datasets.py, covariates.py, windowing.py, scaling.py,
                   toy_pipes.py                                        (§4.1-4.2, §5.1)
      training/    trainer.py                                               (§4.1)
      evaluation/  sampling.py, metrics.py, plots.py                   (§4.3, §5.2)
      utils/       common.py, build.py
    scripts/       train.py, evaluate.py, pipes_experiment.py, run_experiments.py,
                   run_colab_account.py
    configs/       exchange | solar | electricity | traffic | taxi | wikipedia
    tests/         test_flow.py, test_data.py, test_metrics.py, test_backbones.py,
                   test_colab_runner.py

`utils/build.py` is the only place that turns a config into data and a model, so
training and evaluation cannot drift apart.

---

## 3. Milestones and acceptance criteria

| ID | Content | Acceptance criterion | Test |
| --- | --- | --- | --- |
| M0 | config / seed / device, dataset loading | shape matches Table 2, no NaNs | `test_data.py::test_dataset_specs_match_table_2` |
| M1 | unconditional Real NVP + BatchNorm bijection | `max|inverse(forward(x)) - x| < 1e-5`; analytic log-det equals `autograd.functional.jacobian` for D=4; a fitted 2-D Gaussian's covariance is recovered | `test_flow.py` |
| M2 | conditioning on h | two-component mixture with h as the label: samples follow h | `test_flow.py::test_conditioning_is_actually_used` |
| M3 | data layer: windows, covariates, scaling | `invert_scale(apply_scale(x,s),s)==x`; no lag is ever a zero pad; the window sampler stays inside the training range; training covariates equal the ones the sampling loop builds | `test_data.py` |
| M4 | LSTM-Real-NVP end to end | `--overfit-one-window` drives the NLL far down; `h_t` provably does not see `x_t` | `test_backbones.py` |
| M5 | sampling + CRPS / CRPS_sum + fan charts | `--baseline random` scores an order of magnitude worse; energy-form and quantile-form CRPS_sum agree | `test_metrics.py` |
| M6 | toy pipes experiment (§5.1) | the model's covariance and cross-covariance reproduce the data's, and the S_0 -> {S_1,S_2} -> S_3 propagation is positive | `scripts/pipes_experiment.py` |
| M7 | MAF variant | invertibility with the sequential inverse; MADE autoregressivity; permutations present between blocks | `test_flow.py` |
| M8 | Transformer backbone | perturbing a future step leaves every earlier output unchanged | `test_backbones.py::test_transformer_is_causal` |

Order of work: M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> (M6, M7, M8 in any order).
Do not move past M1 until its two tests pass; debugging a flow from inside the
time-series loop is far harder than debugging it standalone.

---

## 4. Data (Appendix A, Table 2)

| dataset | D | domain | freq | total steps | prediction length | source |
| --- | --- | --- | --- | --- | --- | --- |
| Exchange | 8 | R+ | business-daily | 6,071 | 30 | `exchange_rate_nips` |
| Solar | 137 | R+ | hourly | 7,009 | 24 | `solar_nips` |
| Electricity | 370 | R+ | hourly | 5,790 | 24 | `electricity_nips` |
| Traffic | 963 | (0,1) | hourly | 10,413 | 24 | `traffic_nips` |
| Taxi | 1,214 | N | 30-min | 1,488 | 24 | `taxi_30min` |
| Wikipedia | 2,000 | N | daily | 792 | 30 | `wiki2000_nips` |

§5.2: "preprocessed exactly as in Salinas et al. (2019a)" -- that is the GluonTS
`*_nips` (GP-Copula / mv_release) copies. The raw UCI / LSTNet files have
different dimensions (electricity 321, traffic 862) and cannot reproduce Table 2.

Covariates (Appendix D.1): hourly -> hour-of-day, day-of-week, day-of-month with
lags [1, 24, 168]; daily -> day-of-week with lags [1, 7, 14]; half-hourly ->
minute-of-hour, hour-of-day, day-of-week with lags [1, 2, 4, 12, 24, 48].
Time features need **real dates**; lag features are functions of x and are
therefore rebuilt from history, per window in training and per step while
sampling.

Count data (Taxi, Wikipedia): dequantize with Uniform[0,1), then mean-scale.

Split: §5.2 uses rolling windows starting at the last point seen in training.
`train_test_split` cuts `num_test_windows * prediction_length` steps off the end,
training windows are bounded by that cut, and the last `val_windows` prediction
blocks before it are held out to select the best epoch.

---

## 5. Reference numbers

Table 1 (CRPS_sum, mean +/- standard error over 20 reruns) for **Exchange**:

    LSTM-Real-NVP    0.0064 +/- 0.003
    LSTM-MAF         0.005  +/- 0.003
    Transformer-MAF  0.005  +/- 0.003

Those values are normalized: CRPS_sum divided by `sum_t |sum_i x_t^i|`, the
GluonTS `mean_wQuantileLoss` convention on the sum-aggregated series. A raw
CRPS_sum in the units of the data is not comparable with them.
`evaluation/metrics.py` reports the normalized value as `crps_sum`, the raw one as
`crps_sum_raw`, and an independent quantile-based estimate as
`crps_sum_quantile`. Copy the remaining columns of Table 1 (and Table 4 for
marginal CRPS) from the paper before quoting them.

Reproducing the error bars means reruns, not test windows:
`scripts/run_experiments.py --seeds ...` aggregates mean +/- standard error across
runs and writes `outputs/results_table.md`.

---

## 6. Experiments to run

1. `python -m pytest tests -v` -- M1/M2/M3/M5/M7/M8 gates.
2. `python scripts/train.py --config configs/exchange.yaml --overfit-one-window`
   -- M4 gate. Should reach a very low NLL within a few dozen epochs.
3. `python scripts/train.py --config configs/exchange.yaml` then
   `python scripts/evaluate.py --config configs/exchange.yaml --checkpoint checkpoints/exchange_lstm_realnvp_best.pt`
   -- Table 1 row for LSTM-Real-NVP.
4. `python scripts/evaluate.py --config configs/exchange.yaml --baseline random`
   -- M5 sanity check on the metric.
5. `--baseline veclstm` for the Vec-LSTM row; `flow.type: maf` for LSTM-MAF;
   `backbone.type: transformer` for Transformer-MAF.
6. `python scripts/pipes_experiment.py` -- §5.1, Figures 3 and 4.
7. `python scripts/evaluate.py ... --cross-covariance` -- Figure 5.
8. `python scripts/run_experiments.py --config configs/exchange.yaml --seeds 1 2 3`
   -- mean +/- standard error.

---

## 7. Debugging

Symptoms, in the order they usually appear:

1. **NaN loss.** Almost always `exp(s(.))` exploding in the coupling layer, or
   BatchNorm running statistics going bad. `s = tanh(s_raw)` and
   `alpha = tanh(alpha_raw)` are already in place; next steps are lowering the
   learning rate and checking that the input is mean-scaled.
2. **Loss that looks too good.** The backbone is seeing `x_t` while producing
   `h_t`. Run `test_backbones.py::test_lstm_h_t_cannot_see_x_t` and
   `test_transformer_is_causal`.
3. **Loss decreases, samples are noise.** Either a dimension is never
   transformed (alternate the coupling mask; MAF needs permutations between
   blocks) or the conditioning is being ignored (`test_conditioning_is_actually_used`).
4. **Good NLL, terrible CRPS.** The sampling path disagrees with the training
   path: check that the rollout consumes the same cold-start step, and that lag
   covariates during sampling come from the sampled values
   (`test_training_covariates_match_the_sampling_loop`).
5. **CRPS_sum nowhere near 0.0064.** Check the normalization first
   (`crps_sum` vs `crps_sum_raw`), then whether the checkpoint is `_best.pt` or
   the last epoch, then whether training accidentally saw the test range.
6. **Sampling is fine but slow with MAF.** Expected: the MAF inverse is
   sequential, D passes per step (§2.1).

---

## 8. Ours vs the paper's

The paper leaves these open, and every one of them must be flagged as our choice:

* `context_length = prediction_length` (GluonTS convention); §4.1 only says "a
  random time window of size T".
* LSTM `hidden_size = 40`, `num_layers = 2` (Appendix D.2 defers to Salinas et
  al., 2019a).
* `tanh` bounding of the coupling scale and the MAF alpha.
* Gradient clipping at 10.
* Validation split and early stopping (the paper says "max 40 epochs").
* Sinusoidal positional encoding in the Transformer (the paper folds positional
  information into the covariates).
* `num_test_windows` per dataset: taken from the number of rolling evaluations of
  the GluonTS split where that fits, ours for Taxi.
* Scale floor: series that are all zero over the context get scale 1.0.
