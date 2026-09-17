# CAD/USD exchange-rate forecasting

This use case forecasts the FRED `DEXCAUS` series, interpreted as Canadian dollars per U.S. dollar. The first pipeline slice is in `data.py` and registers three cumulative log-return targets:

- `cadusd_logret_1b`: next-business-day movement
- `cadusd_logret_5b`: next-five-business-day movement
- `cadusd_logret_21b`: next-twenty-one-business-day movement

It also registers `wti_log_ret_1b_l1b`, a WTI log-return covariate lagged by one business day. FRED credentials are loaded from the repository-root `.env` as `FRED_API_KEY`.

The recommended experiment order is:

1. Establish naive and traditional statistical baselines.
2. Add structured covariates and compare LightGBM or linear regression.
3. Add the news-aware agent and compare it with the traditional models.

The first two notebooks support this workflow:

- `01_cadusd_data_exploration.ipynb` inspects the target, WTI covariate, and cutoff context.
- `02_cadusd_traditional_backtest.ipynb` compares AutoARIMA, LinearRegression, and LightGBM, with and without WTI.

The baseline smoke experiment is defined in `specs/cadusd_smoke.yaml`. Set
`RUN_BACKTEST = True` in the second notebook to run it. AutoARIMA is the
history-only benchmark; LinearRegression provides an interpretable linear
comparison; LightGBM tests nonlinear relationships and interactions.

The target is a return rather than the exchange-rate level because return distributions are more suitable for comparing forecasts across horizons. A probabilistic forecast should report a point forecast plus quantiles, such as a median and 10th/90th percentile interval.
