# DSE4101: Synthetic Data Augmentation for WTI Crude Oil Price Forecasting

**Python 3.x** | **TensorFlow 2.15.1** | **PyTorch 2.11.0** | **Monthly data: Feb 1986 – Jan 2026**

This study investigates whether synthetic time series augmentation can improve LSTM forecast accuracy for WTI crude oil log-returns. We compare three augmentation methods — **jittering**, **TimeGAN**, and a novel **VAE-TimeGAN** pipeline — against a real-data baseline, tested under both univariate (WTI only) and multivariate (WTI + 7 economic covariates) configurations.

---

## Repository Layout

```
dse4101-project/
├── data/                          # Raw data, processed datasets, augmentation scripts
│   ├── data_extraction.ipynb      # Step 1: Fetch WTI + TB3MS via FRED API
│   ├── predictor_data_extraction.ipynb  # Step 2: Fetch 7 covariates
│   ├── jittering.ipynb            # Jittering augmentation
│   ├── final_data.csv             # Processed real data (committed)
│   ├── final_data_jittered.csv    # Jittered augmented data (committed)
│   ├── final_data_with_VAETIMEGAN.csv   # VAE-TimeGAN augmented data (committed)
│   └── timegan_outputs/           # TimeGAN synthetic data and model checkpoint
│
├── VAE_TimeGAN/                   # VAE + TimeGAN synthesis pipeline
│   ├── data_pipeline.py           # Windowing + MinMaxScaler
│   ├── vae_module.py              # Variational Autoencoder
│   ├── timegan_module.py          # TimeGAN on latent space
│   ├── evaluation_module.py       # Synthetic data quality metrics
│   └── gen_data.ipynb             # End-to-end VAE-TimeGAN training notebook
│
├── Timegan_src/                   # Standalone TimeGAN (raw data path)
│   └── timeGAN_generate_downstream.ipynb
│
├── lstm_src/                      # LSTM training notebooks + results
│   ├── my_lstm.py                 # LSTM architecture and training utilities
│   ├── lstm-1.ipynb               # Univariate baseline
│   ├── lstm-1-jitter.ipynb        # Univariate + jittering
│   ├── lstm-1-timegan.ipynb       # Univariate + TimeGAN
│   ├── lstm-1-vaetimegan.ipynb    # Univariate + VAE-TimeGAN
│   ├── lstm-8.ipynb               # Multivariate baseline (8 vars)
│   ├── lstm-8-jitter.ipynb
│   ├── lstm-8-timegan.ipynb
│   ├── lstm-8-timeganvae.ipynb
│   ├── outofsampleanalysis.ipynb  # Final results and comparison plots
│   └── results/                   # Output CSVs (committed)
│
└── src/                           # Exploratory / legacy notebooks
```

---

## 1. Environment Setup

> **Note:** This project requires both TensorFlow (LSTM) and PyTorch (VAE-TimeGAN). A CUDA-capable GPU is strongly recommended for the generation steps.

```bash
pip install -r requirements.txt
```

**FRED API key** (required for data fetching only — pre-processed data is already committed):

1. Register for a free API key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Create a `.env` file in the project root:
   ```
   FRED_API_KEY=your_key_here
   ```

---

## 2. Data Preparation

> **Skip this section** if using the pre-processed data already committed to the repo (`data/final_data.csv`, `data/predictor_data.csv`).

**Step 1** — Run `data/data_extraction.ipynb`
- Fetches WTI crude oil prices and 3-month Treasury bill rates (TB3MS) from FRED
- Computes WTI log-returns
- Output: `data/final_data.csv` (~480 monthly observations, Feb 1986 – Jan 2026)

**Step 2** — Run `data/predictor_data_extraction.ipynb`
- Fetches 7 macroeconomic/financial covariates: AUD/USD, CAD/USD, NZD/USD, ZAR/USD log-returns, CPI, M1, M2
- Output: `data/predictor_data.csv`

---

## 3. Synthetic Data Generation

> **Skip this section** if using the pre-generated synthetic datasets already committed to the repo. Only required if retraining the generators.

### Method A — Jittering

Notebook: `data/jittering.ipynb`

Adds Gaussian noise (σ = 3% of each column's std) to each 24-month sliding window.

Output: `data/final_data_jittered.csv`

---

### Method B — TimeGAN

Notebook: `Timegan_src/timeGAN_generate_downstream.ipynb`

Trains TimeGAN directly on normalized 24-month windows of the real data.

Outputs:
- `data/timegan_outputs/timegan_stitched_with_real.csv` — real + synthetic combined
- `data/timegan_outputs/timegan_model.pkl` — model checkpoint

---

### Method C — VAE-TimeGAN (novel method)

Notebook: `VAE_TimeGAN/gen_data.ipynb`

This is a two-stage pipeline where a VAE first compresses the multivariate data into a lower-dimensional latent space, then TimeGAN generates synthetic sequences in that latent space, and the VAE decoder reconstructs back to the original feature space.

Pipeline steps:
1. **Data windowing** (`data_pipeline.py`) — builds 24-month sliding windows and fits a MinMaxScaler; saves `VAE_TimeGAN/scaler.pkl`
2. **VAE encoding** (`vae_module.py`) — trains a per-timestep MLP VAE and encodes all windows into latent sequences
3. **TimeGAN generation** (`timegan_module.py`) — trains TimeGAN on latent sequences, generates synthetic latent windows
4. **VAE decoding** — decodes synthetic latent sequences back to data space
5. **Inverse transform** — applies saved scaler to recover original scale

Output: `data/final_data_with_VAETIMEGAN.csv`

Training of the model was conducted in `VAE_TimeGAN/test_ydata.ipynb`. `VAE_TimeGAN/test_custom.ipynb` uses my own custom implementation for TimeGAN.

---

## 4. LSTM Forecasting

Run each notebook in `lstm_src/` to train and evaluate the LSTM under each configuration. Each notebook:
- Performs an expanding-window grid search over hyperparameters on the validation set
- Retrains on train + val with best parameters
- Evaluates on the test set (approx. Feb 2020 – Jan 2026, ~15% of data)
- Saves predictions to `lstm_src/results/`

| Configuration | Notebook |
|---|---|
| Univariate baseline (WTI only) | `lstm-1.ipynb` |
| Univariate + Jittering | `lstm-1-jitter.ipynb` |
| Univariate + TimeGAN | `lstm-1-timegan.ipynb` |
| Univariate + VAE-TimeGAN | `lstm-1-vaetimegan.ipynb` |
| Multivariate baseline (8 vars) | `lstm-8.ipynb` |
| Multivariate + Jittering | `lstm-8-jitter.ipynb` |
| Multivariate + TimeGAN | `lstm-8-timegan.ipynb` |
| Multivariate + VAE-TimeGAN | `lstm-8-timeganvae.ipynb` |

**Architecture:** `LSTM(50) → Dropout → Dense(1)`

**Grid search space:**

| Hyperparameter | Values searched |
|---|---|
| Lookback (months) | 2, 4, 6, 8, 10 |
| Hidden units | 50, 100, 170 |
| Dropout rate | 0.001, 0.01, 0.1 |
| Epochs | 50, 100 |

Best parameters found: `lookback=2, units=50, dropout=0.001, epochs=50`

---

## 5. Out-of-Sample Evaluation & Results

Notebook: `lstm_src/outofsampleanalysis.ipynb`

Reads all result CSVs from `lstm_src/results/` and computes:
- **MSE** (Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (coefficient of determination)

Generates side-by-side forecast plots and a summary comparison table across all 8 configurations.

**Synthetic data quality assessment** (separate from forecast evaluation):

```python
from VAE_TimeGAN.evaluation_module import evaluate_synthetic_data
```

Metrics: statistical moments, autocorrelation (ACF), cross-correlation matrix (Frobenius norm), MMD (Maximum Mean Discrepancy), discriminative score (logistic regression 2-sample test), t-SNE / PCA visualization.

---

## 6. Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Train / val / test split | 70% / 15% / 15% | Temporal order preserved; no data leakage |
| Window length | 24 months | Captures seasonal and business-cycle patterns |
| Scaler | MinMaxScaler [0, 1], fit on train only | Prevents lookahead bias; saved for reuse on synthetic data |
| Validation method | Expanding window (not rolling) | Simulates live forecasting; train set grows over time |
| Forecast horizon | 1-step ahead (monthly) | Matches practical use case |

---

## 7. Acknowledgements

- **Data:** [FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/) via `fredapi`
- **TimeGAN:** Yoon, J., Jarrett, D., & van der Schaar, M. (2019). *Time-series Generative Adversarial Networks.* NeurIPS.
- **ydata-synthetic:** [ydata-ai/ydata-synthetic](https://github.com/ydataai/ydata-synthetic) (used for baseline comparisons)
