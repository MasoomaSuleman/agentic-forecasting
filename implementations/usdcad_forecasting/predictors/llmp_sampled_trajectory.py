"""USD/CAD recipe: sampled-trajectory LLMP (target-only and with-covariates).

This file is intentionally small and explicit so notebook readers can open it
as a reference recipe. The reusable method lives in ``aieng.forecasting``;
this module captures the USD/CAD prompt framing (what the series is and how
exchange-rate returns behave), the default sampling budget, the history
window, and the cache tag used by the experiment.

Two variants share this builder:

- **target-only** — ``covariate_series_ids=None``; the LLM sees only the
  USD/CAD return history.
- **with-covariates** — pass the covariate panel; the predictor serializes
  labeled covariate-history blocks (interest rates, inflation, unemployment,
  oil, USD/CAD returns, etc.) into the prompt. The CRPS gap versus the
  target-only variant answers whether an LLM can exploit the same exogenous
  observations available to the conventional forecasting methods.
"""

from __future__ import annotations

from aieng.forecasting.methods.llm_processes import (
    SampledTrajectoryLLMPredictor,
    SampledTrajectoryLLMPredictorConfig,
)
from aieng.forecasting.models import LITE_MODEL


_DEFAULT_MODEL = LITE_MODEL
_DEFAULT_N_SAMPLES = 10
_DEFAULT_HISTORY_WINDOW = 64
_RECIPE_FAMILY = "usdcad_v1"


_SERIES_DESCRIPTION = (
    "Series: USD/CAD exchange rate (FRED DEXCAUS), represented as the "
    "forward cumulative log return over a fixed number of business days.\n"
    "Interpretation: positive values mean the USD strengthened against the "
    "CAD over the forecast horizon; negative values mean the USD weakened "
    "against the CAD.\n"
    "Units: log-return (a value of 0.01 is roughly a +1% move in USD/CAD).\n"
    "Frequency: business days."
)


_USER_PROMPT_SUFFIX = (
    "Notes for this series:\n"
    "- USD/CAD returns are difficult to predict directionally over short "
    "horizons, so avoid assuming that a recent upward or downward move will "
    "continue without supporting evidence.\n"
    "- Exchange-rate volatility tends to cluster: periods of calm and "
    "periods of elevated volatility can persist. Use recent realised "
    "dispersion as an important guide to the width of the predictive "
    "distribution.\n"
    "- USD/CAD can respond to macroeconomic and financial conditions, "
    "including US-Canada interest-rate differentials, monetary policy "
    "expectations, inflation, labour-market conditions, commodity prices, "
    "and broad USD strength.\n"
    "- Oil prices can be particularly relevant because Canada is a major "
    "commodity and energy exporter, but do not assume a fixed relationship "
    "between oil and USD/CAD in every regime.\n"
    "- Keep the predictive distribution reasonably centered unless the "
    "recent return history or supplied covariate blocks provide clear "
    "evidence for a directional shift.\n"
    "- Distinguish between the direction of the USD/CAD quote and the "
    "strength of the Canadian dollar: a rising USD/CAD means USD strength "
    "relative to CAD, while a falling USD/CAD means CAD strength relative "
    "to USD.\n"
    "- Do not extrapolate a short sequence of positive or negative returns "
    "into a persistent trend without evidence from the broader return "
    "history or covariates."
)


def build_usdcad_llmp_sampled_trajectory(
    *,
    model: str = _DEFAULT_MODEL,
    n_samples: int = _DEFAULT_N_SAMPLES,
    history_window: int | None = _DEFAULT_HISTORY_WINDOW,
    covariate_series_ids: list[str] | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int = 16384,
    variant_tag: str | None = None,
) -> SampledTrajectoryLLMPredictor:
    """Return the USD/CAD sampled-trajectory LLMP recipe.

    The model is a normal parameter because the base LLMP ``predictor_id``
    already includes it. The recipe tag records the USD/CAD prompt/config
    family, whether the covariate panel is in context, and the cache-relevant
    knobs that are not otherwise visible in the ID.

    Parameters
    ----------
    model : str
        Model identifier. Defaults to the lite model
        (``gemini-3.1-flash-lite-preview``).
    n_samples : int
        Number of trajectory samples to draw per prediction call.
    history_window : int or None
        Number of most-recent business days to include in context.
    covariate_series_ids : list[str] or None
        When provided, the covariate panel is serialized into the prompt
        (the "with-covariates" variant). ``None`` is the target-only variant.
    reasoning_effort : str or None
        Provider reasoning budget. ``None`` (default) uses the provider
        default; the Vector proxy rejects ``'disable'``/``'low'``.
    max_tokens : int, default=16384
        Per-call output token budget. The generous default prevents truncation
        on thinking models where thinking tokens consume the same budget via
        the OpenAI-compatible proxy.
    variant_tag : str or None
        Override the cache tag suffix.
    """
    history_tag = (
        "hfull"
        if history_window is None
        else f"h{history_window}"
    )

    sample_count_tag = f"n{n_samples}"

    covariate_tag = (
        "cov"
        if covariate_series_ids
        else "target"
    )

    resolved_variant_tag = (
        variant_tag
        or f"{_RECIPE_FAMILY}_{covariate_tag}_"
        f"{history_tag}_{sample_count_tag}"
    )

    config = SampledTrajectoryLLMPredictorConfig(
        model=model,
        n_samples=n_samples,
        history_window=history_window,
        covariate_series_ids=covariate_series_ids,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        series_description=_SERIES_DESCRIPTION,
        user_prompt_suffix=_USER_PROMPT_SUFFIX,
        variant_tag=resolved_variant_tag,
    )

    return SampledTrajectoryLLMPredictor(config)


__all__ = ["build_usdcad_llmp_sampled_trajectory"]