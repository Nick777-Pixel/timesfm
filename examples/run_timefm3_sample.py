#!/usr/bin/env python3
"""Local TimesFM 3.0 sample runner using realistic test data.

This script keeps the library itself untouched and exercises the public API with
sample time-series inputs drawn from a JSON file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from timesfm3 import ModelConfig, TimesFM3Evaluator


def _as_array(value):
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return arr
    return arr


def _load_cases(path: Path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


def _build_batch(sample_payload):
    contexts = []
    past_only = []
    past_future = []
    ids = []

    for entry in sample_payload["series"]:
        target = _as_array(entry.get("context"))
        po = entry.get("past_only_covariates")
        pf = entry.get("past_future_covariates")
        contexts.append(target)
        past_only.append(_as_array(po) if po is not None else None)
        past_future.append(_as_array(pf) if pf is not None else None)
        ids.append(entry.get("id", f"series_{len(ids)}"))

    return contexts, past_only, past_future, ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local TimesFM 3.0 forecast on sample data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "sample_forecast_cases.json",
        help="Path to the JSON file containing sample forecasting cases.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device to use, e.g. cpu or cuda")
    parser.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch", help="Model checkpoint or HF repo id")
    parser.add_argument("--horizon", type=int, default=None, help="Override forecast horizon for the sample data.")
    args = parser.parse_args()

    payload = _load_cases(args.input)
    defaults = payload.get("defaults", {})
    horizon = int(args.horizon if args.horizon is not None else defaults.get("horizon", 12))

    contexts, po_cov, pf_cov, series_ids = _build_batch(payload)
    config = ModelConfig(
        checkpoint_path=args.checkpoint if args.checkpoint else defaults.get("checkpoint_path", "google/timesfm-3.0-pytorch"),
        per_core_batch_size=2,
        device=args.device if args.device else defaults.get("device", "cpu"),
    )

    forecaster = TimesFM3Evaluator(config)
    outputs = list(
        forecaster.predict_batch(
            contexts=contexts,
            horizon=horizon,
            past_only_covariates=po_cov,
            past_future_covariates=pf_cov,
            ts_ids=series_ids,
            return_quantiles=bool(defaults.get("return_quantiles", True)),
            use_symmetric_averaging=bool(defaults.get("use_symmetric_averaging", False)),
            make_positive=False,
            sort_quantiles=True,
            use_znorm=False,
            padding_mode="none",
            univariate=False,
        )
    )

    print("Forecast results:")
    for result in outputs:
        print(f"- {result.ts_id}: forecast shape={result.forecast.shape}, quantiles shape={result.quantiles.shape}")
        if result.forecast is not None:
            print(f"  forecast values (first 5): {np.asarray(result.forecast)[:5].tolist()}")
        if result.quantiles is not None:
            print(f"  median quantile values (first 5): {np.asarray(result.quantiles)[..., 4][:5].tolist()}")

    print("\nSample workflow complete.")


if __name__ == "__main__":
    main()
