# Upbit Bonus (Crypto) Extension

This folder contains a reproducible pipeline for the course optional bonus:
- Collect Upbit daily OHLCV data (KRW-BTC / KRW-ETH, etc.)
- Build time-series features
- Train a simple ML model
- Convert predictions to allocation weights in [0, 2]
- Enforce the volatility constraint: strategy vol ≤ 1.2 × benchmark vol
- Evaluate with Sharpe-like score, cumulative returns, vol ratio, and max drawdown

## Files
- `upbit_bonus_pipeline.py`: end-to-end script (collection → features → modeling → backtest → artifacts)
- `KRW-BTC_daily_ohlcv.csv`: collected dataset (daily candles)
- `DATASET_CARD_KRW-BTC.md`: dataset card (source, ToS, column definitions)
- `backtest_KRW-BTC_cls.csv`: test-period backtest outputs (allocations, returns, equity curves)
- `metrics_KRW-BTC_cls.json`: metrics summary for the run
- `figure_equity_curve_KRW-BTC_cls.png`: cumulative return/equity curve plot
- `figure_drawdown_KRW-BTC_cls.png`: drawdown plot

## Environment
```bash
python -V
pip install -r requirements.txt
```

`requirements.txt` (minimal):
- numpy
- pandas
- requests
- scikit-learn
- matplotlib

## Reproduce (example)
Run classification (direction) model on BTC daily candles:
```bash
python upbit_bonus_pipeline.py \
  --market KRW-BTC \
  --start 2021-01-01 \
  --end 2025-11-28 \
  --test_days 180 \
  --model cls \
  --annualization 365
```

Outputs are written to `artifacts_upbit_bonus/` by default.

## Example results (KRW-BTC, cls, test_days=180)
- cumulative return (strategy): -0.028809
- cumulative return (benchmark): -0.085620
- sharpe (strategy): -0.015461
- sharpe (benchmark): -0.522089
- vol ratio (strategy/bench): 1.200000
- max drawdown (strategy): 0.285080
- max drawdown (benchmark): 0.284690
- classification accuracy: 0.505556
- AUC: 0.504206

## Notes
- For data collection, this uses Upbit quotation endpoints (public). No API keys are required for daily candles.
- Do not commit exchange API keys to git. If you later extend to authenticated endpoints, use environment variables.
