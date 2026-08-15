# Gold Yahoo Multi-Horizon Forecasting

This folder is a self-contained deployment package. The unchanged model
implementation used by the app is included in `project/`.

The model jointly predicts three aligned daily targets:

```text
return_1d(t) = log(close(t) / close(t-1))
return_3d(t) = log(close(t) / close(t-3))
return_7d(t) = log(close(t) / close(t-7))
```

The next 10 business days are sampled jointly with the project's
Transformer-MAF implementation. The app derives an internally consistent price
path from `return_1d`, renders 80% and 95% price and return fan charts, and shows
daily positive, negative, and threshold-return probabilities for all horizons.

## Train in Colab

1. Upload `fixed_notebooks/project.zip` to
   `/content/drive/MyDrive/fixed_notebooks/project.zip`, or upload it when the
   notebook requests the file.
2. Open `train_gold_yahoo.ipynb` in Google Colab.
3. Select `Runtime > Change runtime type > T4 GPU`.
4. Run all cells.
5. Wait for training and rolling CRPS validation to finish. Validation remains a
   diagnostic; the exported checkpoint always comes from the final epoch.
6. Download the exported files from
   `/content/drive/MyDrive/gold_yahoo_artifacts/`:
   - `gold_yahoo_model.pt`
   - `gold_yahoo_metadata.json`
7. Place both files in `gold-yahoo/artifacts/`.

## Run locally

From `gold-yahoo`:

```text
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of this `gold-yahoo`
   folder, including `project/` and the two trained artifacts.
2. Confirm the checkpoint is below GitHub's per-file size limit. Use Git LFS or
   an external model host if it is larger.
3. Sign in at `share.streamlit.io` with GitHub.
4. Select `Create app` and choose the repository and branch.
5. Keep `requirements.txt` in the repository root; Streamlit Community Cloud
   reads dependency files from the repository root.
6. Set the main file path to `app.py` and deploy.
7. Use `Refresh market data` in the sidebar to clear the 15-minute Yahoo cache
   and recompute all probabilities from the latest daily close.

Training runs only in Colab. Streamlit performs CPU inference and requests only
the short `3mo` Yahoo Finance OHLC window needed to construct the model's
30-trading-day context, seven-day return horizon, and lag features. It does not
download or cache the full Yahoo history. The latest 30 observed candles are
shown beside the 10-day forecast fan. Market data and CPU inference are cached
for 15 minutes, which avoids repeated waits on the free Community Cloud tier.
