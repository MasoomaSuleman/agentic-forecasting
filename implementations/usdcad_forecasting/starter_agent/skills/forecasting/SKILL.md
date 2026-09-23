---
name: forecasting
description: >-
  The output contract for producing a structured probabilistic USD/CAD forecast,
  including quantile rules, calibration guidance, and submission.
---

# Forecasting skill

Load this when the task payload asks for a structured forecast.

## Output contract
1. Produce one forecast per horizon.
2. Use exactly the supplied standard quantile levels.
3. point_forecast must equal the 0.50 quantile.
4. Quantile values must be non-decreasing.
5. Use only information available on or before as_of.
6. Put concise evidence and reasoning in rationale.

Submit with set_model_response using the exact output_schema supplied in the payload.

## USD/CAD interpretation

The target is the forward cumulative log return:
r_t(h) = log(DEXCAUS[t+h] / DEXCAUS[t])

DEXCAUS is Canadian dollars per US dollar. Positive return means USD/CAD rises
and USD strengthens relative to CAD; negative return means CAD strengthens.

Use recent realized volatility to calibrate interval width and widen uncertainty
for longer horizons. Useful drivers include US-Canada rate differentials, BoC
versus Fed expectations, Canadian and US macro data, WTI/commodities, risk
sentiment, and trade/fiscal/geopolitical shocks.
