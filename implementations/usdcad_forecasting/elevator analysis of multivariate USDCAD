# USDCAD Forecasting — Structured Results Analysis

## 1. Executive Summary

This analysis interprets the results produced by `01_usdcad_multivariate_backtest.ipynb`, with emphasis on **what happened** and **why the observed results likely occurred**.

The experiment forecasts **USD/CAD forward cumulative log returns** derived from FRED `DEXCAUS` at three horizons:

- **1 business day (`h=1`)** — next-session return
- **5 business days (`h=5`)** — approximately one-week forward return
- **21 business days (`h=21`)** — approximately one-month forward return

The main comparison contains:

- Naive last-value baseline
- ETS
- Kalman
- AutoARIMA
- Linear Regression
- Linear Regression + 8 covariates
- LightGBM
- LightGBM + 8 covariates
- LLMP using target history only
- LLMP + covariates

The most important result is that **LightGBM + covariates is consistently competitive and is the model evaluated on the protected 2026 window**. Its advantage is especially clear at the 21-business-day horizon.

The results also show that **adding covariates is not automatically beneficial**. The effect depends on both the forecasting horizon and the model's ability to exploit nonlinear relationships. In the smoke backtest, LightGBM benefits substantially from covariates at `h=5` and `h=21`, while the LLMP does not show an improvement from adding the same covariates.

A second important result is methodological: the smoke backtest contains only **5 scored predictions per horizon** (with one skipped origin), so it should be interpreted as a **diagnostic experiment rather than statistically conclusive evidence**. The protected 2026 evaluation provides a more useful check of whether the observed LightGBM + covariate behavior persists outside the development window.

---

# 2. Experimental Setup

## 2.1 Forecast target

The target is not the USD/CAD price level. It is the **forward cumulative log return**:

- `usdcad_logret_1b`
- `usdcad_logret_5b`
- `usdcad_logret_21b`

This distinction matters.

A price-level forecast asks:

> What will USD/CAD be?

The experiment instead asks:

> What cumulative percentage-like movement should we expect over the next 1, 5, or 21 business days?

Using returns reduces the problem of simply learning the long-term level/trend of the exchange rate and focuses the models on **direction, short-term dynamics, and conditional return behavior**.

---

## 2.2 Backtest window

The executed experiment uses:

- **Configuration:** `smoke`
- **Spec:** `usdcad_smoke.yaml`
- **Start:** 2025-10-06
- **End:** 2025-11-14
- **Stride:** 5 business days
- **Warmup:** 250 observations
- **Forecast tasks:** 3
- **LLMP:** enabled because this is a post-cutoff window

The notebook explicitly describes this as a late-2025 weekly-origin smoke test.

The backtest therefore contains a very small number of evaluation origins:

| Horizon | Scores / predictions | Skipped origins |
|---|---:|---:|
| 1 business day | 5 | 1 |
| 5 business days | 5 | 1 |
| 21 business days | 5 | 1 |

This small sample size is one of the most important limitations when interpreting the rankings.

---

# 3. Predictor Design

## 3.1 Conventional models

The conventional comparison contains:

| Model | Target history | Covariates |
|---|---|---|
| Naive | Yes | No |
| ETS | Yes | No |
| Kalman | Yes | No |
| AutoARIMA | Yes | No |
| Linear Regression | Yes | No |
| Linear Regression + covariates | Yes | Yes |
| LightGBM | Yes | No |
| LightGBM + covariates | Yes | Yes |

The regression models use:

- `LAGS = 5`
- `NUM_SAMPLES = 100`

The covariate versions use the same 5 lags for the past covariates.

---

## 3.2 Covariate panel

The experiment successfully registered **8/8 requested covariates**:

1. USD/CAD lagged return
2. Federal funds rate
3. U.S. 2-year Treasury yield
4. U.S. 10-year Treasury yield
5. U.S. 2s10s yield spread
6. CPI month-over-month log difference
7. Unemployment rate
8. Oil return

Conceptually, this gives the models access to three broad types of information:

### FX momentum / autoregressive information

`usdcad_log_ret_1b_l1b`

This captures recent USD/CAD movement.

### Monetary and interest-rate information

- Federal funds rate
- 2-year Treasury yield
- 10-year Treasury yield
- 2s10s spread

These variables provide information about the U.S. monetary-policy and yield environment.

### Macro / commodity information

- CPI
- unemployment
- oil return

These variables represent inflation/labour-market conditions and an important commodity-market signal.

The experiment therefore tests whether USD/CAD returns contain useful information that can be extracted from a broader macro-financial state.

---

# 4. Evaluation Metrics

The main probabilistic metric is **mean CRPS**.

Lower CRPS is better because it rewards forecasts that place probability mass close to the realised outcome while also penalizing poorly calibrated or overly broad predictive distributions.

Direction metrics are also reported:

- Precision for upward movement
- Recall for upward movement
- F1 for upward movement
- Directional accuracy
- ROC-AUC for probability of an upward movement

These metrics answer a different question.

### CRPS asks:

> How good is the entire probabilistic return forecast?

### Directional accuracy asks:

> Did the model correctly identify whether the return was positive or negative?

This distinction is important because a model can have a useful probabilistic forecast even when its directional classification is imperfect, and vice versa.

---

# 5. Backtest Results: What Happened?

## 5.1 Complete smoke-backtest leaderboard

### 1-business-day horizon

| Model | Mean CRPS | Directional Accuracy | ROC-AUC |
|---|---:|---:|---:|
| **LinReg + cov** | **0.00120** | **0.80** | **1.00** |
| LightGBM | 0.00130 | 0.40 | 0.50 |
| ETS | 0.00139 | 0.40 | 0.667 |
| LinReg | 0.00144 | 0.20 | 0.167 |
| LightGBM + cov | 0.00150 | 0.00 | 0.00 |
| Kalman | 0.00150 | 0.40 | 0.50 |
| AutoARIMA | 0.00156 | 0.60 | 0.167 |
| Naive | 0.00158 | 0.60 | 0.583 |
| LLMP (target) | 0.00164 | 0.60 | 0.50 |
| LLMP + cov | 0.00165 | 0.40 | 0.333 |

### What this says

At the one-day horizon, **LinReg + covariates has the lowest CRPS**.

Its CRPS is approximately **24.1% lower than the naive baseline**:

`(0.00158 - 0.00120) / 0.00158 ≈ 24.1%`

It also has:

- 80% directional accuracy
- 1.00 ROC-AUC
- 1.00 upward precision

However, these values are based on only **5 observations**, so they should not be interpreted as evidence of stable 80% directional accuracy.

The more robust interpretation is:

> In this particular late-2025 smoke window, the linear model benefited from the covariate panel at the next-session horizon.

---

# 6. Why Did Covariates Help Linear Regression at h=1?

A likely explanation is that the covariates provided a compact representation of the current macro-financial state.

The linear model assumes that the relationship between lagged predictors and the target can be represented approximately as:

`return_t+1 ≈ β0 + β1 X1 + β2 X2 + ... + βk Xk`

For a short-horizon FX forecast, the signal may be weak but approximately additive over a small local window.

The result suggests that the combination of:

- recent USD/CAD return,
- interest-rate state,
- yield curve,
- inflation,
- unemployment,
- oil,

contained enough information during this particular evaluation period for a linear mapping to outperform the univariate alternatives.

However, the result does **not** demonstrate that each individual covariate was causally responsible. The experiment does not report feature-level ablation or coefficients in the notebook output.

Therefore the appropriate conclusion is:

> The full covariate panel improved the linear model in this backtest; the notebook does not establish which individual covariate caused the improvement.

---

# 7. Five-Day Forecast Results

| Model | Mean CRPS | Directional Accuracy | ROC-AUC |
|---|---:|---:|---:|
| **LightGBM + cov** | **0.00230** | 0.40 | 0.50 |
| Kalman | 0.00237 | 0.60 | 0.50 |
| LightGBM | 0.00242 | 0.40 | 0.667 |
| LinReg + cov | 0.00283 | 0.60 | 0.50 |
| LLMP + cov | 0.00298 | 0.60 | 0.75 |
| LLMP (target) | 0.00303 | 0.60 | 0.50 |
| AutoARIMA | 0.00306 | 0.60 | 0.50 |
| LinReg | 0.00311 | 0.20 | 0.167 |
| ETS | 0.00453 | 0.40 | 0.333 |
| Naive | 0.00506 | 0.40 | 0.417 |

## What happened?

At `h=5`, **LightGBM + covariates produced the lowest CRPS**.

Relative to the naive model:

`(0.00506 - 0.00230) / 0.00506 ≈ 54.5%`

Thus the covariate-enhanced LightGBM reduced CRPS by approximately **54.5% relative to the naive baseline** in the smoke backtest.

The target-only LightGBM also performed strongly:

- LightGBM: 0.00242
- LightGBM + covariates: 0.00230

Adding covariates improved CRPS by approximately **5.0% relative to target-only LightGBM**.

---

# 8. Why Does LightGBM Benefit More at h=5?

This result is consistent with a nonlinear model being able to exploit interactions among macro-financial variables.

A linear model treats each predictor primarily through additive coefficients. LightGBM can represent rules resembling:

> If recent USD/CAD momentum is positive **and** the rate environment is changing **and** oil is moving in a particular direction, adjust the expected return differently.

This is particularly relevant for FX because relationships can be conditional rather than globally linear.

The five-day horizon may also provide more opportunity for macro-financial information to influence the return than a single trading day.

Therefore, one interpretation supported by the experiment is:

> At the one-week horizon, the combination of nonlinear modelling and the covariate panel appears more useful than either the naive forecast or purely univariate models.

But again, the five observations mean this is a **promising pattern, not a statistically established relationship**.

---

# 9. Twenty-One-Day Forecast Results

| Model | Mean CRPS | Directional Accuracy | ROC-AUC |
|---|---:|---:|---:|
| **LightGBM + cov** | **0.00890** | **1.00** | **1.00** |
| AutoARIMA | 0.00996 | 1.00 | 1.00 |
| LinReg | 0.01148 | 0.40 | 0.50 |
| LinReg + cov | 0.01157 | 0.40 | 0.75 |
| LightGBM | 0.01300 | 0.40 | 0.00 |
| ETS | 0.01414 | 0.60 | 0.50 |
| Kalman | 0.01635 | 0.20 | 0.00 |
| LLMP (target) | 0.02081 | 0.40 | 0.00 |
| LLMP + cov | 0.02434 | 0.00 | 0.00 |
| Naive | 0.02451 | 0.40 | 0.625 |

## What happened?

The performance separation becomes much larger at `h=21`.

LightGBM + covariates has:

- CRPS = **0.00890**
- Directional accuracy = **1.00**
- ROC-AUC = **1.00**

Compared with the naive CRPS of 0.02451:

`(0.02451 - 0.00890) / 0.02451 ≈ 63.7%`

So the smoke backtest shows approximately a **63.7% reduction in CRPS relative to naive**.

This is the strongest apparent advantage in the backtest.

---

# 10. Why Does the Covariate Effect Become Stronger at h=21?

This is one of the most interesting findings.

The LightGBM comparison is:

| Horizon | LightGBM | LightGBM + cov | Change |
|---|---:|---:|---:|
| 1 day | 0.00130 | 0.00150 | Worse with covariates |
| 5 days | 0.00242 | 0.00230 | Better with covariates |
| 21 days | 0.01300 | 0.00890 | **Substantially better with covariates** |

The pattern suggests that the covariate panel may be more informative for **medium-horizon cumulative returns** than for immediate next-session movements.

A possible reason is that macro-financial variables generally evolve more slowly than daily FX noise.

For example:

- monetary-policy conditions persist,
- yield relationships persist,
- inflation conditions persist,
- unemployment changes slowly,
- commodity trends can persist across several sessions.

Therefore, a one-month cumulative return may contain more information related to the persistent state of these variables than a single next-day return.

The result should be interpreted as:

> The experiment provides evidence that the usefulness of the covariate panel is horizon-dependent, with the strongest observed LightGBM improvement at 21 business days.

It does **not** prove that the listed macro variables individually predict USD/CAD.

---

# 11. Important Finding: Covariates Do Not Always Help

One of the strongest lessons from the experiment is that:

> More information does not automatically produce a better forecast.

For LightGBM:

- `h=1`: covariates hurt CRPS
- `h=5`: covariates improve CRPS
- `h=21`: covariates substantially improve CRPS

This means the value of a feature depends on:

1. Forecast horizon
2. Model architecture
3. Sample size
4. Signal-to-noise ratio
5. Relationship between predictor and target

At the one-day horizon, adding eight variables may introduce additional noise relative to the very weak next-session signal.

At longer horizons, some of those variables may contain persistent state information that becomes more useful.

This is exactly why a covariate ablation study would be valuable as the next experiment.

---

# 12. LLMP Results: What Happened?

The LLMP comparison is particularly interesting because it tests whether a language-model-based sampled trajectory forecaster can exploit the same information.

## 12.1 LLMP vs LLMP + covariates

| Horizon | LLMP target | LLMP + cov |
|---|---:|---:|
| 1 day | 0.00164 | 0.00165 |
| 5 days | 0.00303 | 0.00298 |
| 21 days | 0.02081 | 0.02434 |

The covariates therefore do **not consistently improve the LLMP**.

At one day:

- Target-only: 0.00164
- With covariates: 0.00165

Essentially no improvement.

At five days:

- Target-only: 0.00303
- With covariates: 0.00298

A small improvement.

At 21 days:

- Target-only: 0.02081
- With covariates: 0.02434

The covariate version is worse.

---

# 13. Why Didn't the LLMP Benefit Like LightGBM?

This is an important research result.

The same eight covariates that helped LightGBM substantially at the 21-day horizon did not produce the same benefit for the LLMP.

A plausible explanation is that **having information available is different from extracting predictive structure from that information**.

LightGBM explicitly optimizes a numerical prediction function from structured features. Its training procedure can discover nonlinear partitions and interactions.

The LLMP receives the covariate history through prompt blocks. It must infer numerical relationships from textual/numerical context.

That creates several possible difficulties:

### 13.1 Numerical precision

Financial prediction often depends on relatively small numerical differences. Language models are not inherently optimized as numerical regression estimators.

### 13.2 Weak signal

The relationship between macro variables and short-term FX returns is likely weak relative to market noise.

A language model may have difficulty converting weak numerical correlations into calibrated probability distributions.

### 13.3 Covariate overload

Adding eight covariates increases the amount of information the model must interpret.

If the model cannot reliably distinguish relevant signals from irrelevant variation, more context can make the forecast less precise.

### 13.4 Prompt representation

The covariates are represented as labelled historical blocks rather than as a learned feature representation.

LightGBM receives a structured feature matrix.

The LLMP receives a prompt.

That difference is central to interpreting the result.

---

# 14. LLMP vs Gradient-Boosted Models

The smoke results show a clear gap between LLMP and LightGBM at longer horizons.

At `h=21`:

- LightGBM + cov: **0.00890**
- LightGBM: **0.01300**
- LLMP target: **0.02081**
- LLMP + cov: **0.02434**

The LLMP + covariate model has a CRPS approximately **2.74×** the LightGBM + covariate CRPS:

`0.02434 / 0.00890 ≈ 2.74`

This suggests that, in this experiment, the numerical ML model is substantially better at converting the structured covariate panel into a calibrated return distribution.

The important scientific conclusion is not simply that one model is "better".

The more useful observation is:

> **The benefit of structured numerical covariates is model-dependent.**

The same information can produce different forecasting performance depending on how the forecasting architecture processes it.

---

# 15. Protected 2026 Evaluation

The notebook then evaluates only two finalists:

- Naive
- LightGBM + covariates

This is important because the 2026 window is explicitly treated as a protected evaluation.

## 15.1 Results

| Horizon | Model | CRPS | Directional Accuracy | ROC-AUC |
|---|---|---:|---:|---:|
| 1 day | **LightGBM + cov** | **0.00138** | 0.500 | 0.533 |
| 1 day | Naive | 0.00426 | 0.500 | 0.533 |
| 5 days | **LightGBM + cov** | **0.00462** | 0.571 | 0.600 |
| 5 days | Naive | 0.00784 | 0.429 | 0.300 |
| 21 days | **LightGBM + cov** | **0.00688** | 0.625 | 0.875 |
| 21 days | Naive | 0.01932 | 0.375 | 0.375 |

---

# 16. What Does the 2026 Evaluation Tell Us?

The most important observation is that the LightGBM + covariate model retains a substantial CRPS advantage over the naive baseline on the protected evaluation.

### h=1

LightGBM + cov:

`0.00138`

Naive:

`0.00426`

Relative CRPS reduction:

`(0.00426 - 0.00138) / 0.00426 ≈ 67.6%`

### h=5

LightGBM + cov:

`0.00462`

Naive:

`0.00784`

Relative CRPS reduction:

`≈ 41.1%`

### h=21

LightGBM + cov:

`0.00688`

Naive:

`0.01932`

Relative CRPS reduction:

`≈ 64.4%`

So the protected evaluation preserves the central pattern seen in the smoke test:

> **The covariate-enhanced LightGBM produces substantially lower probabilistic forecast error than the naive baseline across all three horizons.**

This is more important than the exact smoke-test ranking because the 2026 window is outside the model-development period.

---

# 17. Why Is the 2026 Result Important?

The smoke backtest is small enough that a strong result could simply be due to a particular sequence of observations.

The 2026 evaluation tests whether the same model configuration maintains its advantage on a different period.

The fact that LightGBM + covariates still has lower CRPS at:

- 1 day
- 5 days
- 21 days

provides evidence that the observed backtest behavior is not limited entirely to the original smoke window.

However, the 2026 evaluation is also small:

- 8 predictions at `h=1`
- 7 predictions at `h=5`
- 8 predictions at `h=21`

Therefore, the result should still be described as **promising evidence of generalization**, rather than a definitive estimate of live trading performance.

---

# 18. Directional Results: An Important Nuance

The CRPS improvement is more consistent than the directional metrics.

For example, at `h=1` in the protected evaluation:

| Model | CRPS | Directional Accuracy | ROC-AUC |
|---|---:|---:|---:|
| LightGBM + cov | 0.00138 | 0.50 | 0.533 |
| Naive | 0.00426 | 0.50 | 0.533 |

The models have identical directional metrics, despite the large CRPS difference.

This is important.

It means the LightGBM advantage is not necessarily coming from simply predicting the correct sign more often.

Instead, its predictive distribution can be substantially better calibrated or closer to the realised magnitude even when the direction classification is unchanged.

This is a strong argument for keeping **probabilistic metrics such as CRPS** rather than evaluating the system only using directional accuracy.

---

# 19. The h=21 Directional Signal Is Especially Interesting

At 21 business days in the protected evaluation:

- LightGBM + cov directional accuracy = **0.625**
- Naive directional accuracy = **0.375**
- LightGBM + cov ROC-AUC = **0.875**
- Naive ROC-AUC = **0.375**

The model therefore shows both:

1. Lower probabilistic forecast error
2. Better directional discrimination

This combination is more informative than the smoke-test h=21 result alone.

Still, only 8 evaluation cases are available, so the numerical values should not be treated as stable estimates of future directional accuracy.

---

# 20. Overall Pattern Across Horizons

The experiment reveals a useful horizon-dependent pattern.

| Horizon | Main result | Interpretation |
|---|---|---|
| **1 day** | Linear Regression + cov best in smoke; LightGBM + cov strong in protected eval | Very noisy short-term signal; covariates can help but benefit is model-sensitive |
| **5 days** | LightGBM + cov best in smoke and protected eval | Nonlinear model + macro/market state provides useful information |
| **21 days** | LightGBM + cov clearly best | Persistent macro-financial state appears more useful at longer horizon |

The key pattern is:

> **The value of the covariate panel increases as the forecasting horizon moves from immediate next-session movement toward one-month cumulative movement, particularly for LightGBM.**

This should be tested with a larger backtest before being treated as a general property of USD/CAD.

---

# 21. Why the Naive Baseline Becomes Relatively Weak at Longer Horizons

The naive model carries forward the most recent return information.

That can be useful when short-term return persistence exists.

But as the forecast horizon grows, simply extending the recent return becomes increasingly difficult.

At 21 business days, the target represents an accumulated movement across an entire month.

A single most-recent-return observation is unlikely to encode the complete state of:

- interest rates,
- monetary policy,
- inflation,
- labour market conditions,
- oil,
- recent FX momentum.

The larger CRPS gap between Naive and LightGBM + cov at 21 days is therefore consistent with the richer feature set being more useful for a longer-horizon target.

---

# 22. What the Results Do NOT Prove

The notebook results should not be interpreted as proving any of the following:

### Not proven: individual causal effects

The experiment does not show that oil, CPI, unemployment, or a particular yield independently causes USD/CAD returns.

### Not proven: stable directional accuracy

The reported directional metrics are calculated on only 5 smoke observations and 7–8 protected-evaluation observations.

### Not proven: trading profitability

The notebook evaluates forecasting metrics.

It does not report:

- transaction costs,
- bid/ask spreads,
- slippage,
- position sizing,
- leverage,
- turnover,
- drawdown,
- Sharpe ratio,
- P&L.

Therefore forecast improvement is not equivalent to demonstrated trading profitability.

### Not proven: universal superiority of LightGBM

The result is specific to this dataset, feature panel, forecast setup, model configuration, and evaluation windows.

### Not proven: LLMs cannot forecast FX

The LLMP result is specific to the tested sampled-trajectory implementation, model, prompt representation, history window, and eight-sample configuration.

---

# 23. Main "What and Why" Findings

## Finding 1 — Structured numerical ML is effective

**What:** LightGBM + covariates consistently produces low CRPS and remains strong on the protected 2026 evaluation.

**Why:** The model can learn nonlinear relationships and interactions among lagged FX returns and macro-financial variables.

---

## Finding 2 — Covariates are horizon-dependent

**What:** Covariates hurt LightGBM at `h=1`, help slightly at `h=5`, and help substantially at `h=21` in the smoke backtest.

**Why:** Very short-horizon FX movements are noisy, while persistent macro-financial conditions may have more influence on cumulative multi-week returns.

---

## Finding 3 — More features do not automatically improve forecasting

**What:** Adding eight covariates does not consistently improve every model.

**Why:** Additional variables only help if the forecasting method can identify and exploit their predictive structure without adding excessive noise.

---

## Finding 4 — LLMP does not automatically benefit from numerical context

**What:** LLMP + covariates does not outperform target-only LLMP consistently and is substantially weaker than LightGBM + covariates at 21 days.

**Why:** The LLM receives numerical information through prompt context rather than through a learned structured feature representation optimized for regression.

---

## Finding 5 — CRPS reveals information that direction metrics miss

**What:** At h=1 in the protected evaluation, LightGBM + cov and Naive have identical directional accuracy and ROC-AUC, but LightGBM + cov has much lower CRPS.

**Why:** A probabilistic forecast can be better calibrated or closer in magnitude without changing the predicted sign.

---

## Finding 6 — The protected evaluation supports the main development result

**What:** LightGBM + cov retains a large CRPS advantage over Naive in 2026.

**Why:** The model's learned relationship appears to transfer beyond the late-2025 smoke period, although the protected sample remains small.

---

# 24. Recommended Interpretation for a Project Report

A concise research interpretation could be:

> The USDCAD experiments show that forecast performance is strongly horizon- and model-dependent. In the late-2025 smoke backtest, LightGBM with the eight-variable macro-financial covariate panel achieved the lowest CRPS at the 5- and 21-business-day horizons, while linear regression with covariates performed best at the one-day horizon. The value of covariates was not uniform: they slightly degraded LightGBM performance at one day but improved it at five days and substantially at 21 days, suggesting that structured macro-financial information may be more useful for medium-horizon cumulative FX returns than for highly noisy next-session movements. The LLMP did not obtain the same benefit from covariates, indicating that providing numerical context to an LLM is not equivalent to learning structured nonlinear relationships from numerical features. Most importantly, the selected LightGBM + covariate model retained a substantial CRPS advantage over the naive baseline on the protected 2026 evaluation across all three horizons. However, because the smoke and protected evaluation windows contain only a small number of forecast origins, the results should be viewed as evidence of promising model behavior rather than statistically conclusive evidence of stable forecasting or trading performance.

---

# 25. Recommended Next Experiments

## 25.1 Run the full 2025 backtest

The notebook already provides:

`EXPERIMENT_CONFIG = "backtest_2025"`

The full 2025 experiment is described as approximately 50 weekly origins.

This should be the next major validation step.

Why:

- substantially larger sample size,
- more market conditions,
- less sensitivity to individual observations,
- more reliable model ranking.

---

## 25.2 Perform a covariate ablation study

Instead of comparing only:

`target-only`

vs.

`all 8 covariates`

test groups such as:

### FX-only

- USD/CAD lagged return

### Rates-only

- Federal funds
- U.S. 2Y
- U.S. 10Y
- 2s10s spread

### Macro-only

- CPI
- unemployment

### Commodity-only

- oil

### Full panel

- all eight variables

This would answer the missing "why" question more directly:

> **Which information actually contributes to the forecast improvement?**

---

# 26. Recommended Interaction Analysis

Because LightGBM performs particularly well with covariates, inspect feature importance and/or SHAP values.

The objective should not only be:

> Which feature is most important?

It should be:

> Which features matter at which forecast horizon?

For example, compare feature importance separately for:

- h=1
- h=5
- h=21

A particularly interesting hypothesis from the current results is:

> Rate/yield and commodity information may become relatively more important at longer horizons, while recent USD/CAD return dynamics may dominate shorter horizons.

This is a hypothesis to test, not a conclusion established by the current notebook.

---

# 27. Recommended Robustness Tests

The next iteration should test:

1. More forecast origins
2. Multiple market regimes
3. Rolling/expanding training windows
4. Different lag lengths
5. Different LightGBM hyperparameters
6. Covariate ablations
7. Feature permutation tests
8. Statistical significance / confidence intervals
9. Performance by volatility regime
10. Performance during large USD/CAD moves

A particularly useful experiment would divide observations into:

- low-volatility periods
- high-volatility periods

and compare CRPS separately.

This would answer whether the covariate advantage is concentrated in particular market regimes.

---

# 28. Recommended LLMP Follow-up

The current LLMP experiment uses:

- history window = 48
- sampled trajectories = 8
- target-only vs covariate prompt
- Gemini-based LLMP implementation

The next question should be whether the weak LLMP result is caused by the model itself or by the prompt representation.

Test:

1. Different history windows
2. Different number of sampled trajectories
3. Normalized rather than raw numerical inputs
4. Explicit feature summaries
5. Returns instead of levels
6. Separate rate/commodity/macro sections
7. Explicit trend and volatility summaries
8. Structured JSON/tabular prompt format
9. Few-shot examples
10. A numerical baseline with exactly the same information

The most scientifically useful comparison is:

> **same information → different representation → different forecasting architecture**

---

# 29. Important Reproducibility Note

The notebook itself contains several inherited references to **"S&P 500"** in titles/comments even though the executed implementation is clearly configured for **USDCAD**:

- `usdcad_forecasting`
- `DEXCAUS`
- `usdcad_logret_*`
- USDCAD-specific predictor IDs
- USDCAD covariates

Before using this notebook as a final project artifact, those stale S&P 500 references should be renamed to USDCAD.

This is a documentation issue rather than a forecasting-result issue, but fixing it will make the experiment much easier to reproduce and review.

---

# 30. Final Research Takeaway

The strongest evidence from the current experiment is not simply that one model has the lowest number.

The more useful result is a **three-part pattern**:

1. **Forecast horizon matters.**
   Short-term USD/CAD returns are difficult and noisy, while longer cumulative-return forecasts appear to benefit more from macro-financial context.

2. **Model architecture matters.**
   The same eight covariates that help LightGBM substantially at longer horizons do not consistently help the LLMP.

3. **Probabilistic quality matters beyond direction.**
   CRPS reveals improvements that directional accuracy alone would miss.

The protected 2026 evaluation strengthens the central finding: **LightGBM + covariates maintains a substantial probabilistic-forecasting advantage over the naive baseline across 1-, 5-, and 21-business-day horizons.**

At the same time, the current number of evaluation origins is too small to claim stable real-world forecasting performance. The next decisive experiment is therefore the **full 2025 backtest with substantially more origins**, followed by **covariate ablation and feature-level analysis**.

The resulting research question becomes:

> **Which macro-financial signals provide genuinely transferable information for USD/CAD returns, at which forecast horizons, and which forecasting architectures can reliably extract that information?**

That is the key question that the current results motivate.
