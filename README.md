# TimesFM

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation
model developed by Google Research for time-series forecasting.

*   Paper:
    [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688),
    ICML 2024.
*   <span style="color:red">(NEW!)</span> TimesFM 3.0 Checkpoint:
    [`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch).
*   Checkpoints (up to 2.5):
    [TimesFM Hugging Face Collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6).
*   [Google Research blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
    (New blog post for TimesFM 3.0 coming soon!).
*   TimesFM in Google 1P Products:
    *   [BigQuery ML](https://cloud.google.com/bigquery/docs/timesfm-model):
        Enterprise level SQL queries for scalability and reliability.
    *   [Google Sheets](https://workspaceupdates.googleblog.com/2026/02/forecast-data-in-connected-sheets-BigQueryML-TimesFM.html):
        For your daily spreadsheet.
    *   [Vertex Model Garden](https://pantheon.corp.google.com/vertex-ai/publishers/google/model-garden/timesfm):
        Dockerized endpoint for agentic calling.

This open version is not an officially supported Google product.

**Latest Model Version:** TimesFM 3.0

**Archived Model Versions:**

-   2.5: relevant code under `src/timesfm`.
-   1.0 and 2.0: relevant code archived in the subdirectory `v1`. You can `pip
    install timesfm==1.3.0` to install an older version of this package to load
    them.

--------------------------------------------------------------------------------

## Update — August 2026

**TimesFM 3.0 is out!**

TimesFM 3.0 introduces native **multivariate time-series forecasting**, flexible
**covariate support** (both past-only and past-and-future covariates), superior
zero-shot generalist capabilities, and top performance across all three major
time-series foundation model benchmarks.

### Key Highlights:

-   **Native Multivariate & Univariate Forecasting with Covariates**: Seamlessly
    forecast multi-channel multivariate series as well as individual univariate
    series, with native support for past-only and past-and-future dynamic
    covariates without per-task tuning.
-   **Top Benchmark Performance**:
    -   🥇 **fev-bench**: **Rank #1 overall** across 100 diverse real-world
        forecasting tasks.
    -   🥇 **TIME Benchmark**: **Rank #1 overall** across 50 domain datasets and
        98 evaluation tasks.
    -   🥇 **GIFT-Eval**: **Rank #1 among all foundation models**.

### License notice for pretrained weights

> **Important:** The TimesFM source code in this repository is licensed under
> Apache-2.0, and model weights up to version 2.5 remain Apache-2.0. However,
> for the time being, TimesFM 3.0 pretrained weights are distributed under the
> separate `timesfm-non-commercial-license-v1.0` license and are restricted to
> non-commercial, non-production use. Commercial or production use of the
> default pretrained weights is **not permitted**.

--------------------------------------------------------------------------------

## Update - July 2, 2026

Updated PyPI to `timesfm=2.0.2`. See
[Install](https://github.com/google-research/timesfm#from-pypi).

## Update - Apr. 9, 2026

Added fine-tuning example using HuggingFace Transformers + PEFT (LoRA) — see
[`timesfm-forecasting/examples/finetuning/`](timesfm-forecasting/examples/finetuning/).
Also added unit tests (`tests/`) and incorporated several community fixes.

Shoutout to [@kashif](https://github.com/kashif) and
[@darkpowerxo](https://github.com/darkpowerxo).

## Update - Mar. 19, 2026

Huge shoutout to [@borealBytes](https://github.com/borealBytes) for adding the
support for
[AGENTS](https://github.com/google-research/timesfm/blob/master/AGENTS.md)!
TimesFM
[SKILL.md](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting)
is out.

## Update - Oct. 29, 2025

Added back the covariate support through XReg for TimesFM 2.5.

## Update - Sept. 15, 2025

TimesFM 2.5 is out!

Comparing to TimesFM 2.0, this new 2.5 model:

-   uses 200M parameters, down from 500M.
-   supports up to 16k context length, up from 2048.
-   supports continuous quantile forecast up to 1k horizon via an optional 30M
    quantile head.
-   gets rid of the `frequency` indicator.
-   has a couple of new forecasting flags.

Since the Sept. 2025 launch, the following improvements have been completed for
TimesFM 2.5:

1.  ✅ Flax version of the model for faster inference.
2.  ✅ Covariate support via XReg (see Oct. 2025 update).
3.  ✅ Documentation, examples, and agent skill (see `timesfm-forecasting/`).
4.  ✅ Fine-tuning example with LoRA via HuggingFace Transformers + PEFT (see
    `timesfm-forecasting/examples/finetuning/`).
5.  ✅ Unit tests for core layers, configs, and utilities (see `tests/`).

### Install

#### From `PyPI`

```shell
# Install TimesFM with PyTorch
pip install timesfm[torch]
```

#### Local Install

1.  Clone the repository:

    ```shell
    git clone https://github.com/google-research/timesfm.git
    cd timesfm
    ```

2.  Create a virtual environment and install with PyTorch:

    ```shell
    # Using uv
    uv venv
    source .venv/bin/activate

     # Install the package in editable mode with torch
    uv pip install -e .[torch]
    ```

--------------------------------------------------------------------------------

## Table of Contents

-   [Install](#install)
-   [Code Examples: TimesFM 3.0](#code-examples-timesfm-30)
-   [Understanding Forecast Results](#understanding-forecast-results)
-   [Update — August 2026](#update--august-2026)
-   [Update - July 2, 2026](#update---july-2-2026)
-   [Update - Apr. 9, 2026](#update---apr-9-2026)
-   [Update - Mar. 19, 2026](#update---mar-19-2026)
-   [Update - Oct. 29, 2025](#update---oct-29-2025)
-   [Update - Sept. 15, 2025](#update---sept-15-2025)

--------------------------------------------------------------------------------

## Understanding Forecast Results

TimesFM produces a forecast for the future horizon only. The historical values you pass into the model are treated as input context, while the returned array represents the model's prediction for the next steps.

The project currently exposes the following result fields through the public API:

-   `ts_id`: the optional identifier for each series.
-   `forecast`: a point forecast for each future step.
-   `quantiles`: a full quantile forecast for each future step.
-   `forecast.shape`: the horizon length for a univariate output.
-   `quantiles.shape`: the horizon length by the number of quantiles.

### Reading the terminal output

The sample workflow prints one result per series. For example, the project output currently looks like this:

```text
Forecast results:
- series_1: forecast shape=(12,), quantiles shape=(12, 9)
  forecast values (first 5): [191.95196533203125, 194.01124572753906, 195.78428649902344, 197.26284790039062, 198.37374877929688]
  median quantile values (first 5): [191.95196533203125, 194.01124572753906, 195.78428649902344, 197.26284790039062, 198.37374877929688]
- series_2: forecast shape=(12,), quantiles shape=(12, 9)
  forecast values (first 5): [161.9519805908203, 164.01129150390625, 165.78436279296875, 167.26295471191406, 168.37388610839844]
  median quantile values (first 5): [161.9519805908203, 164.01129150390625, 165.78436279296875, 167.26295471191406, 168.37388610839844]
```

This represents a 12-step forecast for each time series. The values are not the historical context; they are the model's predicted future values for the next 12 periods.

### Forecasted values

`forecast` is the model's point prediction for each future step. In the sample output above, the first forecasted value for `series_1` is `191.95`, the second is `194.01`, and so on. These are the primary values to use when evaluating the direction and magnitude of the upcoming horizon.

For the default configuration in this project, `forecast` corresponds to the median quantile prediction, which is the 5th element of the `quantiles` bundle. In other words, the model is producing a point estimate and a set of uncertainty quantiles together.

### Historical vs. predicted values

The project does not print the historical input series alongside the forecast in the sample terminal output. Instead, the history is the time series you provide as the `context` argument, and the forecast is generated for the future window. A typical evaluation workflow is:

1. Load historical values.
2. Run `predict_batch(...)` or `predict(...)`.
3. Compare the historical trend with the forecasted horizon.
4. Assess whether the future values continue the observed trend, flatten, rise, or decay.

The forecast should be interpreted as future behavior beyond the last observed timestamp, not as a reconstruction of historical data.

### Confidence and prediction intervals

The project returns quantiles for each step, with a shape of `(horizon, 9)` in the sample run. The default quantile set is the standard 9-point family from `0.1` to `0.9`.

In this output, the `quantiles` array contains the distribution of possible outcomes for each forecast step. The median is the middle quantile (`[..., 4]`). The other quantiles provide the lower and upper uncertainty bands:

-   lower quantiles: roughly `0.1`, `0.2`, `0.3`, `0.4`
-   median: `0.5`
-   upper quantiles: roughly `0.6`, `0.7`, `0.8`, `0.9`

A wider spread between lower and upper quantiles usually indicates higher uncertainty. A narrower spread usually indicates a more confident prediction. Use the quantile range to judge how certain the forecast is at each step.

### Model accuracy and error metrics

The sample terminal workflow currently reports forecast values and quantiles, but it does not compute or print standard accuracy metrics such as MAE, RMSE, MAPE, or coverage. Those metrics are not produced by the sample runner itself.

When evaluating a forecast, the user should compare the model output against known actual future values (ground truth) using their own evaluation code. Common metrics include:

-   MAE: mean absolute error
-   RMSE: root mean squared error
-   MAPE: mean absolute percentage error
-   Quantile loss or coverage: how often actual values fall inside the model's predicted intervals

This project's current output is primarily a forecasting artifact, not a full benchmark report.

### Trends, seasonality, and anomalies

The output should be interpreted qualitatively as well as quantitatively:

-   Trends: does the forecast continue an upward, downward, or flat trajectory?
-   Seasonality: does the pattern repeat at a regular interval or show periodic structure?
-   Anomalies: were there sudden changes or outlier events in the historical context that are not captured in the future forecast? The model does not explicitly label anomalies; you must inspect the forecast and the original series together.

A forecast with a steadily rising median and reasonably narrow quantile bands usually suggests a stable trend. A forecast with wide quantile spreads or abrupt changes may indicate uncertainty or potential regime shifts.

### Other outputs currently produced by the project

The project currently emits the following useful metadata in the sample terminal output:

-   `ts_id`: identifies each series in the batch.
-   `forecast shape`: the horizon length for each series.
-   `quantiles shape`: the horizon-by-quantile matrix.
-   `median quantile values`: the center estimate for each step.

The runner is not currently printing downstream analytics like error rates, summary tables, or anomaly flags. For those, the user should add post-processing around the returned `forecast` and `quantiles` arrays.

### How to use the result in practice

When evaluating a TimesFM forecast, treat the output as follows:

-   Use `forecast` as the main point estimate.
-   Use `quantiles` to judge uncertainty.
-   Compare the forecast horizon against the historical context to assess plausibility.
-   Evaluate actual future performance using external ground truth and metrics such as MAE or RMSE.
-   Inspect trend, seasonality, and anomalies visually or through custom analysis.

This is exactly how the project's sample workflow behaves today: it produces a forecast and quantile distribution, but the user is expected to interpret those values and optionally add their own downstream evaluation logic.

--------------------------------------------------------------------------------

### Code Examples: TimesFM 3.0

#### 1. Univariate Forecasting (Variable Lengths)

Pass a batch of 1D NumPy arrays of different context lengths to forecast
univariate time series:

```python
import numpy as np
from timesfm3 import TimesFM3Evaluator, ModelConfig

# Initialize TimesFM 3.0
config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=32,
    device="cuda"
)
forecaster = TimesFM3Evaluator(config)

# Two univariate series of different lengths (100 and 72 steps)
ts1 = np.linspace(0, 1, 100).astype(np.float32)
ts2 = np.sin(np.linspace(0, 24, 72)).astype(np.float32)

# Generate forecast (point predictions + 9 quantiles: 0.1 to 0.9)
outputs = list(forecaster.predict_batch([ts1, ts2], horizon=12, return_quantiles=True, use_symmetric_averaging=False))

print("Series 1 forecast shape:", outputs[0].forecast.shape)   # (12,)
print("Series 1 quantiles shape:", outputs[0].quantiles.shape) # (12, 9)

print("Series 2 forecast shape:", outputs[1].forecast.shape)   # (12,)
print("Series 2 quantiles shape:", outputs[1].quantiles.shape) # (12, 9)
```

#### 2. Multivariate Forecasting with Covariates

Pass a 2D array of shape `(num_variates, context_length)` along with optional
past-only and past-and-future covariates:

```python
import numpy as np
from timesfm3 import TimesFM3Evaluator, ModelConfig

# Initialize TimesFM 3.0
config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=16,
    device="cuda"
)
forecaster = TimesFM3Evaluator(config)

context_len = 128
horizon = 24

# 3 target variates across past context: (3, 128)
target = np.random.randn(3, context_len).astype(np.float32)

# 1 past-only covariate channel across past context: (1, 128)
past_only_cov = np.random.randn(1, context_len).astype(np.float32)

# 2 past-and-future covariate channels across context + horizon: (2, 152)
past_future_cov = np.random.randn(2, context_len + horizon).astype(np.float32)

# Generate joint forecast across all 3 target variates
outputs = list(
    forecaster.predict_batch(
        contexts=[target],
        horizon=horizon,
        past_only_covariates=[past_only_cov],
        past_future_covariates=[past_future_cov],
        return_quantiles=True,
        use_symmetric_averaging=False,
    )
)

print("Multivariate forecast shape:", outputs[0].forecast.shape)   # (3, 24)
print("Multivariate quantiles shape:", outputs[0].quantiles.shape) # (3, 24, 9)
```
