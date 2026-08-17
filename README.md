# Gold Yahoo multi-model forecasting

Self-contained deployment package. `project/` is the verbatim paper
implementation and is never edited; `goldflow/` is the gold-specific layer that
the Colab notebook and the Streamlit app both import, so training and inference
cannot drift apart.

## The four models

| Key | Model | Backbone | Head |
| --- | --- | --- | --- |
| `lstm_realnvp` | LSTM-Real-NVP | LSTM | Real-NVP coupling flow |
| `lstm_maf` | LSTM-MAF | LSTM | masked autoregressive flow |
| `transformer_maf` | Transformer-MAF | attention | masked autoregressive flow |
| `veclstm` | Vec-LSTM-ind-scaling | LSTM | diagonal Gaussian |

The first three are the paper's own variants. Vec-LSTM is the baseline control:
same backbone and objective, but its head factorizes over the three horizons, so
whatever the flows win on `CRPS_sum` is attributable to modelling the joint
cross-section rather than to capacity.

On top of those, two boosting stages:

- **Stage 1** forward stagewise CRPS selection. Each round adds the model that
  most reduces the `CRPS_sum` of the mixture, with replacement. The result is a
  weighted mixture of the four flows, i.e. a real predictive distribution scored
  by the same metric as the individual models.
- **Stage 2** a gradient-boosted residual correction of the mixture median. It
  is fitted on stride-1 windows inside the validation range, gated on a disjoint
  tail of that range, and only exported if it beats the mixture by more than
  `booster_min_gain_pct`. If it does not, the app serves stage 1 alone and the
  tables still show what stage 2 would have done.

Ensemble weights and the residual booster never touch the test range, so the
reported test metrics are honest for the boosted rows too.

## The target and what gets reported

Each day is a three-dimensional vector of trailing log returns:

```text
return_1d(t) = log(close(t) / close(t-1))
return_3d(t) = log(close(t) / close(t-3))
return_7d(t) = log(close(t) / close(t-7))
```

Every horizon window ends on day `t`, so day `t` is always inside it:
`return_3d(t) = r(t-2) + r(t-1) + r(t)`. That is also exactly the close-to-close
return of a real 3-day candle whose last daily candle is day `t`, which is what
`goldflow.data.aggregate_candles` builds for the app's candle charts. Cell 3 of
the notebook asserts the two agree.

**Reported horizon returns are reconstructed, not read off the network.** The
flow samples all three dimensions freely, so its raw `return_3d` for forecast
day 1 ignores that two of those three days have already traded. Instead:

1. the daily dimension of a sample is compounded onto the last observed close,
2. that price path is spliced onto the observed closes,
3. every horizon is read off the spliced series:
   `return_h at day k = log(P(t+k) / P(t+k-h))`, where any index at or before `t`
   is a real close.

So the 3-day return on forecast day 1 is the sampled daily return plus two known
closes, the numbers are mutually consistent with the price fan by construction,
and `P(return > 0)` refers to a quantity that actually settles. The raw
dimensions are still shown, under "Model calibration", because their divergence
from the reconstruction on the *late* forecast days measures how well the joint
distribution was learned.

## Train in Colab

1. Build the bundle locally and upload it:

   ```text
   python tools/make_colab_bundle.py
   ```

   Upload `gold_yahoo_colab.zip` to
   `/content/drive/MyDrive/fixed_notebooks/gold_yahoo_colab.zip`.

2. Open `train_gold_yahoo.ipynb` in Colab and select
   `Runtime > Change runtime type > T4 GPU`.
3. Run all cells. The default budget is four variants times three seeds, which
   is twelve training runs and roughly an hour on a T4. Set `seeds=(1,)` in cell
   2 for a fast pass; the tables then report a single run with no standard error.
4. The last cell copies everything to `/content/drive/MyDrive/gold_yahoo_artifacts/`
   and downloads a zip. Unpack it over `artifacts/` so it contains:

   ```text
   artifacts/gold_yahoo_metadata.json
   artifacts/models/{lstm_realnvp,lstm_maf,transformer_maf,veclstm}.pt
   artifacts/experiments/*.csv
   artifacts/figures/*.png
   artifacts/gold_yahoo_booster.joblib     # only if stage 2 passed its gate
   ```

Checkpoints from before this version cannot be reused. The calendar covariates
changed (see below), so old weights were conditioned on different features and
the app rejects them.

## Reproduced experiments

The notebook reproduces the paper's experiments section for `GC=F`:

| Export | Paper | Metric |
| --- | --- | --- |
| `table1_crps_sum.csv` | Table 1 | test set `CRPS_sum` |
| `table4_crps.csv` | Table 4 | test set `CRPS` (Appendix B.2) |
| `table5_mse.csv` | Table 5 | test set `MSE` (Appendix B.3) |

Orientation matches the paper: columns are models, rows are the data set.
Because gold is a single data set, `CRPS` and `MSE` are also broken out per
return horizon — the appendix averages those over `D`, so this is the same
quantity at finer granularity. `CRPS_sum` is defined on the sum across `D` and
stays a single row. Entries are `mean±stderr` over the seeds, matching "the mean
and standard errors are obtained by re-running each method three times".

Two aggregation details are copied from `project/scripts/evaluate.py`:
`CRPS_sum` and `CRPS` are GluonTS `mean_wQuantileLoss` values aggregated as a
ratio of global sums rather than a mean of per-window ratios, and the mean-scaler
fallback is pooled across evaluation windows before any window is scored.

Also exported: `leaderboard.csv` (all metrics, ranked, with the gap to the best
model), `coverage.csv` (empirical 80%/95% interval coverage, because a model can
win `CRPS_sum` and still have bands too tight to size a position with),
`per_window_metrics.csv`, `per_seed.csv`, `per_horizon.csv`, plus paper-style
figures: prediction intervals (Fig. 8-11) and cross-covariance heatmaps (Fig. 5).

## Run locally

```text
pip install -r requirements.txt
streamlit run app.py
```

The dashboard has one tab per model, a **Boosted ensemble** tab, an
**Experiments** tab with the reproduced tables and the winner, and a
**Market data** tab with the real daily, 3-day and 7-day candles. Every tab
carries its own price fan, return fan, probability table and calibration report.

## Deploy to Streamlit Community Cloud

1. Push this folder, including `project/`, `goldflow/` and the trained artifacts.
2. The four checkpoints are a few MB each. Use Git LFS if your repository limits
   per-file size.
3. Sign in at `share.streamlit.io`, select `Create app`, and pick the repository.
4. Keep `requirements.txt` at the repository root and set the main file to
   `app.py`.
5. Use `Refresh market data` in the sidebar to clear the 15-minute cache.

Training happens only in Colab. Streamlit does CPU inference and downloads a
short `1y` daily window — enough for the 30-day context, the two lags and the
7-day horizon — not the full Yahoo history. Market data and inference are cached
for 15 minutes.

## Fixed: calendar covariates depended on how much history was loaded

`project/flowforecast/data/covariates.py` derives its Fourier period from the
array it is given (`period = max(values.max()) + 1`). That is fine for the
paper's datasets, where training and evaluation see the same array, but the
notebook built features over the whole Yahoo history (`dayofyear` reaching 366)
while the app only downloaded a short window (`dayofyear` reaching perhaps 230).
The same calendar day therefore got a different phase at training and at
inference time, so the app was feeding the network covariates it had never been
trained on.

`goldflow/features.py` fixes every period to a constant, so the features depend
only on the timestamp. The channel layout is unchanged, `cond_size` is still 10,
and the spec is recorded in the exported metadata: the app refuses to run against
a checkpoint whose covariate definition differs. `project/` was not modified.

## Verification

Run any of these locally; none needs a GPU or Colab, and only `--live` needs
network access.

```text
python tools/smoke_test.py            # horizon maths, candles, covariates, all 4 variants
python tools/run_pipeline_local.py    # the whole Colab pipeline, tiny budget
python tools/check_notebook.py        # every notebook cell compiles and resolves
python tools/check_notebook_cells.py  # executes the notebook's real cell sources
python tools/check_app.py             # app loading, forecasting, tables
python tools/check_app_render.py      # renders every tab through Streamlit AppTest
```
