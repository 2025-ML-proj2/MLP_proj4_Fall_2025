# MLP_proj4_Fall_2025
## How to run 'Step 1~5'
1. Upload or write the each code in the Kaggle Notebook environment.
2. This code recommends to GPU P100 or T4 x2, so verifiying phone number to access GPU is required. However, it can use CPU if necessary.
3. Internet connection in Kaggle Notebook->Settings is not required.
5. Set the Settings -> Accelerator to GPU.
6. Run the code and submit the results.

## Upbit Bonus (Crypto) Pipeline
- Collect daily OHLCV from Upbit Quotation API (public)
- Build features (lags/rolling/EMA-inspired)
- Train a simple ML model (classification or regression)
- Convert predictions to allocation in [0, 2]
- Enforce volatility constraint: vol(strategy) <= 1.2 * vol(benchmark)
- Evaluate: Sharpe-like score, cumulative return, vol ratio, max drawdown
- Write a Dataset Card and results artifacts for reproducibility

Usage examples:
```
  python upbit_bonus_pipeline.py --market KRW-BTC --start 2021-01-01 --end 2025-11-28 --test_days 180 --model cls
```
```
  python upbit_bonus_pipeline.py --market KRW-ETH --start 2021-01-01 --end 2025-11-28 --model reg --annualization 365
```

Notes:
- This script intentionally uses only public quotation endpoints; no Access/Secret keys are required.
- If you later add private endpoints (balances/orders), store keys in environment variables, never hardcode.