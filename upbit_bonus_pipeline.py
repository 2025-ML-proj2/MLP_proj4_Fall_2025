
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upbit Bonus (Crypto) Pipeline
- Collect daily OHLCV from Upbit Quotation API (public)
- Build features (lags/rolling/EMA-inspired)
- Train a simple ML model (classification or regression)
- Convert predictions to allocation in [0, 2]
- Enforce volatility constraint: vol(strategy) <= 1.2 * vol(benchmark)
- Evaluate: Sharpe-like score, cumulative return, vol ratio, max drawdown
- Write a Dataset Card and results artifacts for reproducibility

Usage examples:
  python upbit_bonus_pipeline.py --market KRW-BTC --start 2021-01-01 --end 2025-11-28 --test_days 180 --model cls
  python upbit_bonus_pipeline.py --market KRW-ETH --start 2021-01-01 --end 2025-11-28 --model reg --annualization 365

Notes:
- This script intentionally uses only public quotation endpoints; no Access/Secret keys are required.
- If you later add private endpoints (balances/orders), store keys in environment variables, never hardcode.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


UPBIT_BASE_URL = "https://api.upbit.com"
DAY_CANDLES_PATH = "/v1/candles/days"

# Upbit Candle group rate limit: up to 10 calls/sec (IP basis). Use conservative spacing.
MIN_SECONDS_BETWEEN_CALLS = 0.12


@dataclasses.dataclass(frozen=True)
class BacktestResult:
    metrics: Dict[str, float]
    results_df: pd.DataFrame


def _iso_kst_midnight(date_str_yyyy_mm_dd: str) -> str:
    # to parameter expects ISO8601 with timezone, e.g. 2025-07-30T00:00:00+09:00
    return f"{date_str_yyyy_mm_dd}T00:00:00+09:00"


def fetch_upbit_day_candles(
    market: str,
    start: str,
    end: str,
    *,
    base_url: str = UPBIT_BASE_URL,
    sleep_s: float = MIN_SECONDS_BETWEEN_CALLS,
) -> pd.DataFrame:
    """
    Fetch daily candles between [start, end] inclusive by paging backwards with 'to'.
    Upbit returns candles in reverse chronological order, max count=200 per call.
    """
    url = base_url.rstrip("/") + DAY_CANDLES_PATH

    end_dt = pd.to_datetime(end).date()
    start_dt = pd.to_datetime(start).date()
    if start_dt > end_dt:
        raise ValueError("start must be <= end")

    # start paging from the day after end (because 'to' is exclusive)
    cursor_dt = end_dt + pd.Timedelta(days=1)

    all_rows: List[dict] = []
    last_call = 0.0

    while True:
        elapsed = time.time() - last_call
        if elapsed < sleep_s:
            time.sleep(sleep_s - elapsed)

        params = {
            "market": market,
            "to": _iso_kst_midnight(cursor_dt.strftime("%Y-%m-%d")),
            "count": 200,
        }
        resp = requests.get(url, params=params, headers={"accept": "application/json"}, timeout=30)
        last_call = time.time()

        if resp.status_code != 200:
            raise RuntimeError(f"Upbit API error: {resp.status_code} {resp.text[:300]}")

        batch = resp.json()
        if not batch:
            break

        all_rows.extend(batch)

        kst_times = [pd.to_datetime(x["candle_date_time_kst"]) for x in batch if "candle_date_time_kst" in x]
        if not kst_times:
            break
        earliest = min(kst_times).date()
        cursor_dt = pd.to_datetime(earliest)

        if earliest <= start_dt:
            break

    if not all_rows:
        raise RuntimeError("No candle data returned. Check market code and date range.")

    df = pd.DataFrame(all_rows)
    df["date_kst"] = pd.to_datetime(df["candle_date_time_kst"]).dt.date
    df = df.rename(
        columns={
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
            "candle_acc_trade_price": "value",
        }
    )

    keep_cols = ["market", "date_kst", "open", "high", "low", "close", "volume", "value"]
    df = df[keep_cols].copy()

    df = df.sort_values("date_kst").reset_index(drop=True)
    df = df[(df["date_kst"] >= start_dt) & (df["date_kst"] <= end_dt)].reset_index(drop=True)
    df["date_id"] = np.arange(len(df), dtype=int)
    df = df.drop_duplicates(subset=["date_kst"], keep="last").reset_index(drop=True)

    return df


def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    d = df.copy()
    d["date_kst"] = pd.to_datetime(d["date_kst"])

    d["ret_1"] = d["close"].pct_change()
    d["log_ret_1"] = np.log(d["close"]).diff()
    d["hl_range"] = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
    d["oc_return"] = (d["close"] - d["open"]) / d["open"].replace(0, np.nan)
    d["vol_chg"] = d["volume"].pct_change()
    d["val_chg"] = d["value"].pct_change()

    for lag in [1, 2, 3, 5, 10]:
        d[f"ret_lag_{lag}"] = d["ret_1"].shift(lag)
        d[f"vol_chg_lag_{lag}"] = d["vol_chg"].shift(lag)
        d[f"hl_range_lag_{lag}"] = d["hl_range"].shift(lag)

    for w in [5, 10, 20, 60]:
        d[f"ret_roll_mean_{w}"] = d["ret_1"].rolling(w).mean()
        d[f"ret_roll_std_{w}"] = d["ret_1"].rolling(w).std()
        d[f"vol_roll_mean_{w}"] = d["volume"].rolling(w).mean()
        d[f"hl_roll_mean_{w}"] = d["hl_range"].rolling(w).mean()

    for span in [10, 30, 60]:
        d[f"ret_ewm_{span}"] = d["ret_1"].ewm(span=span, adjust=False).mean()
        d[f"vol_ewm_{span}"] = d["volume"].ewm(span=span, adjust=False).mean()

    y_reg = d["ret_1"].shift(-1)
    y_cls = (y_reg > 0).astype(int)
    bench_next = d["ret_1"].shift(-1)

    exclude = {
        "market", "date_kst", "date_id",
        "open", "high", "low", "close", "volume", "value",
        "ret_1", "log_ret_1",
    }
    feature_cols = [c for c in d.columns if c not in exclude]
    X = d[feature_cols].copy()

    return X, y_reg, y_cls, bench_next


def prediction_to_allocation_from_proba(
    p_up: np.ndarray,
    *,
    dead_zone: float = 0.02,
    confidence: float = 4.0,
) -> np.ndarray:
    p_up = np.clip(p_up, 0.0, 1.0)
    edge = p_up - 0.5
    alloc = np.ones_like(p_up, dtype=float)
    mask = np.abs(edge) >= dead_zone
    alloc[mask] = 1.0 + np.tanh(confidence * edge[mask])
    return np.clip(alloc, 0.0, 2.0)


def prediction_to_allocation_from_return(
    r_hat: np.ndarray,
    *,
    dead_zone: float = 0.0005,
    confidence: float = 250.0,
) -> np.ndarray:
    r_hat = np.asarray(r_hat, dtype=float)
    alloc = np.ones_like(r_hat, dtype=float)
    mask = np.abs(r_hat) >= dead_zone
    alloc[mask] = 1.0 + np.tanh(confidence * r_hat[mask])
    return np.clip(alloc, 0.0, 2.0)


def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd = 1.0 - (equity_curve / np.maximum(peak, 1e-12))
    return float(np.max(dd))


def annualized_sharpe(daily_returns: np.ndarray, annualization: int = 365) -> float:
    r = np.asarray(daily_returns, dtype=float)
    mu = np.nanmean(r)
    sig = np.nanstd(r, ddof=1)
    if not np.isfinite(sig) or sig == 0:
        return 0.0
    return float((mu / sig) * math.sqrt(annualization))


def realized_vol(daily_returns: np.ndarray) -> float:
    r = np.asarray(daily_returns, dtype=float)
    return float(np.nanstd(r, ddof=1))


def enforce_vol_constraint(
    alloc: np.ndarray,
    bench_ret: np.ndarray,
    *,
    max_ratio: float = 1.2,
    n_iter: int = 30,
) -> Tuple[np.ndarray, float]:
    alloc = np.asarray(alloc, dtype=float)
    bench_ret = np.asarray(bench_ret, dtype=float)

    target_vol = max_ratio * realized_vol(bench_ret)
    if target_vol <= 0:
        return np.clip(alloc, 0.0, 2.0), 1.0

    def vol_for_k(k: float) -> float:
        a = 1.0 + k * (alloc - 1.0)
        a = np.clip(a, 0.0, 2.0)
        strat_ret = a * bench_ret
        return realized_vol(strat_ret)

    if vol_for_k(1.0) <= target_vol:
        return np.clip(alloc, 0.0, 2.0), 1.0

    lo, hi = 0.0, 1.0
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        if vol_for_k(mid) <= target_vol:
            lo = mid
        else:
            hi = mid
    k_star = lo
    alloc_adj = 1.0 + k_star * (alloc - 1.0)
    return np.clip(alloc_adj, 0.0, 2.0), float(k_star)


def backtest(
    df_raw: pd.DataFrame,
    *,
    model_kind: str = "cls",
    test_days: int = 180,
    annualization: int = 365,
    seed: int = 42,
) -> BacktestResult:
    np.random.seed(seed)

    X, y_reg, y_cls, bench_next = make_features(df_raw)

    valid_mask = y_reg.notna() & bench_next.notna()
    X = X[valid_mask].reset_index(drop=True)
    y_reg = y_reg[valid_mask].reset_index(drop=True)
    y_cls = y_cls[valid_mask].reset_index(drop=True)
    bench_next = bench_next[valid_mask].reset_index(drop=True)

    n = len(X)
    if n < (test_days + 60):
        raise ValueError(f"Not enough data after feature engineering. Need at least ~{test_days+60} rows, got {n}.")

    split = n - test_days
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train_reg, y_test_reg = y_reg.iloc[:split], y_reg.iloc[split:]
    y_train_cls, y_test_cls = y_cls.iloc[:split], y_cls.iloc[split:]
    bench_test = bench_next.iloc[split:].to_numpy()

    if model_kind == "cls":
        model = HistGradientBoostingClassifier(
            learning_rate=0.07, max_depth=4, max_iter=300, random_state=seed
        )
    elif model_kind == "reg":
        model = HistGradientBoostingRegressor(
            learning_rate=0.07, max_depth=4, max_iter=300, random_state=seed
        )
    else:
        raise ValueError("--model must be one of: cls, reg")

    pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("model", model)])

    if model_kind == "cls":
        pipe.fit(X_train, y_train_cls)
        p_up = pipe.predict_proba(X_test)[:, 1]
        alloc_raw = prediction_to_allocation_from_proba(p_up)

        y_pred = (p_up >= 0.5).astype(int)
        acc = accuracy_score(y_test_cls, y_pred)
        try:
            auc = roc_auc_score(y_test_cls, p_up)
        except Exception:
            auc = float("nan")
        rmse = float("nan")
    else:
        pipe.fit(X_train, y_train_reg)
        r_hat = pipe.predict(X_test)
        alloc_raw = prediction_to_allocation_from_return(r_hat)

        acc = float("nan")
        auc = float("nan")
        rmse = float(math.sqrt(mean_squared_error(y_test_reg, r_hat)))

    alloc_adj, k_star = enforce_vol_constraint(alloc_raw, bench_test, max_ratio=1.2)

    strat_ret = alloc_adj * bench_test
    bench_ret = 1.0 * bench_test

    eq_strat = np.cumprod(1.0 + np.nan_to_num(strat_ret, nan=0.0))
    eq_bench = np.cumprod(1.0 + np.nan_to_num(bench_ret, nan=0.0))

    vol_strat = realized_vol(strat_ret)
    vol_bench = realized_vol(bench_ret)
    vol_ratio = float(vol_strat / vol_bench) if vol_bench > 0 else float("inf")

    sharpe_strat = annualized_sharpe(strat_ret, annualization=annualization)
    sharpe_bench = annualized_sharpe(bench_ret, annualization=annualization)

    mdd_strat = max_drawdown(eq_strat)
    mdd_bench = max_drawdown(eq_bench)

    metrics = {
        "n_total": float(n),
        "test_days": float(test_days),
        "model": model_kind,
        "vol_constraint_k": float(k_star),
        "vol_bench": float(vol_bench),
        "vol_strat": float(vol_strat),
        "vol_ratio": float(vol_ratio),
        "sharpe_strat": float(sharpe_strat),
        "sharpe_bench": float(sharpe_bench),
        "mdd_strat": float(mdd_strat),
        "mdd_bench": float(mdd_bench),
        "acc_cls": float(acc) if np.isfinite(acc) else float("nan"),
        "auc_cls": float(auc) if np.isfinite(auc) else float("nan"),
        "rmse_reg": float(rmse) if np.isfinite(rmse) else float("nan"),
        "cum_return_strat": float(eq_strat[-1] - 1.0),
        "cum_return_bench": float(eq_bench[-1] - 1.0),
    }

    # Align test dates (we lose one day due to pct_change and one due to shift(-1))
    # The backtest rows correspond to days where we had a next-day return.
    valid_dates = pd.to_datetime(df_raw["date_kst"]).iloc[1:].reset_index(drop=True)
    valid_dates = valid_dates[valid_mask.reset_index(drop=True)].reset_index(drop=True)
    test_dates = valid_dates.iloc[split:].reset_index(drop=True)

    out = pd.DataFrame(
        {
            "date_kst": test_dates,
            "bench_ret_next": bench_ret,
            "allocation_raw": alloc_raw,
            "allocation_adj": alloc_adj,
            "strategy_ret": strat_ret,
            "benchmark_equity": eq_bench,
            "strategy_equity": eq_strat,
        }
    )

    return BacktestResult(metrics=metrics, results_df=out)


def write_dataset_card(out_path: str, market: str, start: str, end: str) -> None:
    card = f"""# Dataset Card: Upbit Daily Candles ({market})

## 1) Source
- Provider: Upbit Open API (Quotation REST API)
- Endpoint: `GET https://api.upbit.com/v1/candles/days`
- Documentation: https://docs.upbit.com/kr/reference/list-candles-days

## 2) Collection method
- Method: Official REST API (no web scraping)
- Pagination: Pull 200 rows per call and page backward using the `to` parameter (exclusive).
- Date range requested: {start} to {end}

## 3) Rate limits and compliance
- Candle group is rate-limited on a per-second basis; implement request spacing (sleep) and handle 429 responses.
- Reference: https://docs.upbit.com/kr/reference/rate-limits

## 4) License / Terms of Service (ToS)
- Data and content provided via Open API are subject to Upbit Open API Terms.
- Reference: https://static.upbit.com/terms/legacy/openapi_agreement_20231215.html

## 5) Columns (raw)
- `date_kst`: Candle date (KST)
- `open`, `high`, `low`, `close`: Daily OHLC
- `volume`: Accumulated trade volume
- `value`: Accumulated trade value
- `date_id`: Sequential integer index (added for convenience)
- `market`: Market code (e.g., KRW-BTC)

## 6) Derived columns (features / labels)
- Features: lagged returns, rolling mean/std volatility proxies, EWM statistics, range/volume dynamics
- Label (regression): next-day close-to-close return
- Label (classification): 1 if next-day return > 0 else 0
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(card)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", type=str, default="KRW-BTC")
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts_upbit_bonus")
    ap.add_argument("--test_days", type=int, default=180)
    ap.add_argument("--model", type=str, choices=["cls", "reg"], default="cls")
    ap.add_argument("--annualization", type=int, default=365)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = fetch_upbit_day_candles(args.market, args.start, args.end)
    raw_csv = os.path.join(args.out_dir, f"{args.market}_daily_ohlcv.csv")
    df.to_csv(raw_csv, index=False)

    write_dataset_card(os.path.join(args.out_dir, f"DATASET_CARD_{args.market}.md"), args.market, args.start, args.end)

    bt = backtest(df, model_kind=args.model, test_days=args.test_days, annualization=args.annualization, seed=args.seed)

    metrics_path = os.path.join(args.out_dir, f"metrics_{args.market}_{args.model}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(bt.metrics, f, indent=2, ensure_ascii=False)

    results_path = os.path.join(args.out_dir, f"backtest_{args.market}_{args.model}.csv")
    bt.results_df.to_csv(results_path, index=False)

    print("\n=== Upbit Bonus Pipeline Summary ===")
    print(f"Market: {args.market} | Period: {args.start} to {args.end}")
    print(f"Artifacts in: {os.path.abspath(args.out_dir)}")
    print("\nMetrics:")
    for k, v in bt.metrics.items():
        if isinstance(v, float):
            print(f"  {k:>18s}: {v:.6f}")
        else:
            print(f"  {k:>18s}: {v}")
    print("\nSaved:")
    print(f"  - Raw dataset: {raw_csv}")
    print(f"  - Dataset card: {os.path.join(args.out_dir, f'DATASET_CARD_{args.market}.md')}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Backtest: {results_path}\n")


if __name__ == "__main__":
    main()
