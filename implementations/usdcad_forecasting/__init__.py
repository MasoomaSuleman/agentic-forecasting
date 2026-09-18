"""USD/CAD multivariate log-return forecasting experiment.

The demo notebooks are narrative shells over the modules in this directory:

- :mod:`data` — ``build_usdcad_multivariate_service()`` and canonical
  USD/CAD/FRED covariate ids.
- :mod:`predictors` — forecasting predictor implementations.
- :mod:`leaderboard` — ``build_leaderboard()`` turns cached results into
  ``RESULTS_DF``.
- :mod:`analysis` — styled leaderboards and directional metrics.
- :mod:`plots` — matplotlib figures for target history, forecast performance,
  and realised-vs-predicted returns.

The primary target is the FRED USD/CAD exchange-rate series ``DEXCAUS``.
Forward log-return targets are available at 1-, 5-, and 21-business-day
horizons.

See ``README.md`` for the full experiment description.
"""

from .data import (
    DEFAULT_COVARIATE_SERIES_IDS,
    FRED_PREFETCH_REGISTRY,
    FRED_SERIES_IDS_FOR_PREFETCH,
    SERIES_ID_CPI,
    SERIES_ID_FED_FUNDS,
    SERIES_ID_OIL_RETURN,
    SERIES_ID_USDCAD_RETURN,
    SERIES_ID_UNEMPLOYMENT,
    SERIES_ID_US_10Y,
    SERIES_ID_US_2Y,
    SERIES_ID_US_2Y10Y_SPREAD,
    USDCAD_LOG_RETURN_SERIES_ID,
    USDCAD_RETURN_TARGETS,
    USDCAD_RETURN_WINDOWS,
    USDCAD_SERIES_ID,
    build_usdcad_log_return_service,
    build_usdcad_multivariate_service,
    usdcad_logret_series_id,
)

__all__ = [
    "DEFAULT_COVARIATE_SERIES_IDS",
    "FRED_PREFETCH_REGISTRY",
    "FRED_SERIES_IDS_FOR_PREFETCH",
    "SERIES_ID_CPI",
    "SERIES_ID_FED_FUNDS",
    "SERIES_ID_OIL_RETURN",
    "SERIES_ID_USDCAD_RETURN",
    "SERIES_ID_UNEMPLOYMENT",
    "SERIES_ID_US_10Y",
    "SERIES_ID_US_2Y",
    "SERIES_ID_US_2Y10Y_SPREAD",
    "USDCAD_LOG_RETURN_SERIES_ID",
    "USDCAD_RETURN_TARGETS",
    "USDCAD_RETURN_WINDOWS",
    "USDCAD_SERIES_ID",
    "build_usdcad_log_return_service",
    "build_usdcad_multivariate_service",
    "usdcad_logret_series_id",
]