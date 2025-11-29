# Dataset Card: Upbit Daily Candles (KRW-BTC)

## 1) Source
- Provider: Upbit Open API (Quotation REST API)
- Endpoint: `GET https://api.upbit.com/v1/candles/days`
- Documentation: https://docs.upbit.com/kr/reference/list-candles-days

## 2) Collection method
- Method: Official REST API (no web scraping)
- Pagination: Pull 200 rows per call and page backward using the `to` parameter (exclusive).
- Date range requested: 2021-01-01 to 2025-11-28

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
