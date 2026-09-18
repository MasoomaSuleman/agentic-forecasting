"""Leak-safe FRED data service for USD/CAD exchange-rate forecasting.

Target:
    USD/CAD exchange rate from FRED series ``DEXCAUS``.

Targets are forward cumulative log returns:

    r^(N)_t = log(P[t+N] / P[t])

where N is the forecasting horizon in business days.

Registered targets:
    - usdcad_logret_1b   -> next-business-day return
    - usdcad_logret_5b   -> forward 5-business-day return
    - usdcad_logret_21b  -> forward 21-business-day return

FRED covariates:
    - US effective federal funds rate
    - US 2-year Treasury yield
    - US 10-year Treasury yield
    - US 2Y-10Y yield spread
    - US CPI month-over-month log change
    - US unemployment rate
    - WTI oil price return
    - USD/CAD lagged return

Anti-leakage policy:
    - Daily covariates are lagged by one business day.
    - Monthly macroeconomic variables are expanded from conservative
      release dates before being lagged.
    - DataService cutoff handling prevents unavailable observations
      from entering forecasting contexts.

All raw data are retrieved through FREDAdapter and cached under:

    data/fred/{FRED_SERIES_ID}.parquet

Run the repository's FRED fetch script first if required:

    uv run python scripts/fetch_fred.py

The FRED API key is loaded from the repository-root ``.env`` file.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters import FREDAdapter
from aieng.forecasting.data.features import StaticFrameAdapter
from aieng.forecasting.data.features import (
    apply_one_business_day_feature_lag as _apply_one_business_day_feature_lag,
)
from aieng.forecasting.data.features import (
    business_daily_expand_from_releases as _business_daily_expand_from_releases,
)
from aieng.forecasting.data.features import (
    business_daily_ffill as _business_daily_ffill,
)
from aieng.forecasting.data.features import (
    canonical_three_col as _canonical_three_col,
)
from aieng.forecasting.data.features import (
    drop_weekend_timestamp_rows as _drop_weekend_timestamp_rows,
)

_load_dotenv: Callable[..., Any] | None

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None


# ---------------------------------------------------------------------------
# Repository / environment helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path | None:
    """Find the repository root."""
    here = Path(__file__).resolve()

    for path in (here, *here.parents):
        if (path / "aieng-forecasting").is_dir():
            return path

    return None


def _load_fred_dotenv() -> None:
    """Load FRED_API_KEY from the repository-root .env file."""
    if _load_dotenv is None:
        return

    root = _repo_root()

    if root is not None:
        _load_dotenv(root / ".env", override=False)


def _default_cache_dir() -> Path:
    """Return the repository FRED cache directory."""
    root = _repo_root()

    if root is not None:
        return root / "data" / "fred"

    return Path("data") / "fred"


DEFAULT_FRED_CACHE_DIR = _default_cache_dir()


# ---------------------------------------------------------------------------
# USD/CAD target configuration
# ---------------------------------------------------------------------------

# FRED:
# DEXCAUS = Canadian Dollars to One U.S. Dollar
USDCAD_SERIES_ID = "DEXCAUS"

# Forecasting horizons in business days.
USDCAD_RETURN_WINDOWS: tuple[int, ...] = (1, 5, 21)

USDCAD_WINDOW_LABELS: dict[int, str] = {
    1: "next-session",
    5: "forward 1-week (5 business days)",
    21: "forward 1-month (21 business days)",
}


def usdcad_logret_series_id(window: int) -> str:
    """Return the canonical USD/CAD target id for a horizon."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")

    return f"usdcad_logret_{window}b"


USDCAD_RETURN_TARGETS: dict[int, str] = {
    window: usdcad_logret_series_id(window)
    for window in USDCAD_RETURN_WINDOWS
}

# Canonical daily target.
USDCAD_LOG_RETURN_SERIES_ID = usdcad_logret_series_id(1)


# ---------------------------------------------------------------------------
# FRED covariates
# ---------------------------------------------------------------------------

# FRED series used by this implementation.
FRED_PREFETCH_REGISTRY: dict[str, tuple[str, str, str]] = {
    "DEXCAUS": (
        "Canadian Dollars to One U.S. Dollar",
        "Canadian dollars per U.S. dollar",
        "D",
    ),
    "DFF": (
        "Effective Federal Funds Rate",
        "Percent",
        "D",
    ),
    "DGS2": (
        "2-Year Treasury Constant Maturity Rate",
        "Percent",
        "D",
    ),
    "DGS10": (
        "10-Year Treasury Constant Maturity Rate",
        "Percent",
        "D",
    ),
    "CPIAUCSL": (
        "Consumer Price Index for All Urban Consumers: All Items",
        "Index 1982-84=100",
        "MS",
    ),
    "UNRATE": (
        "Unemployment Rate",
        "Percent",
        "MS",
    ),
    "DCOILWTICO": (
        "Crude Oil Prices: West Texas Intermediate (WTI)",
        "Dollars per Barrel",
        "D",
    ),
}


FRED_SERIES_IDS_FOR_PREFETCH: tuple[str, ...] = (
    tuple(FRED_PREFETCH_REGISTRY.keys())
)


# ---------------------------------------------------------------------------
# Covariate series IDs
# ---------------------------------------------------------------------------

SERIES_ID_USDCAD_RETURN = "usdcad_log_ret_1b_l1b"
SERIES_ID_FED_FUNDS = "fed_funds_level_l1b"
SERIES_ID_US_2Y = "ust2y_level_l1b"
SERIES_ID_US_10Y = "ust10y_level_l1b"
SERIES_ID_US_2Y10Y_SPREAD = "ust2y10y_spread_l1b"
SERIES_ID_CPI = "cpi_mom_logdiff_l1b"
SERIES_ID_UNEMPLOYMENT = "unemployment_rate_l1b"
SERIES_ID_OIL_RETURN = "oil_log_ret_1b_l1b"


DEFAULT_COVARIATE_SERIES_IDS: list[str] = [
    SERIES_ID_USDCAD_RETURN,
    SERIES_ID_FED_FUNDS,
    SERIES_ID_US_2Y,
    SERIES_ID_US_10Y,
    SERIES_ID_US_2Y10Y_SPREAD,
    SERIES_ID_CPI,
    SERIES_ID_UNEMPLOYMENT,
    SERIES_ID_OIL_RETURN,
]


# ---------------------------------------------------------------------------
# Generic FRED loader
# ---------------------------------------------------------------------------


def _fred_frame(
    fred_id: str,
    *,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    """Load a FRED series and normalize it to timestamp/value/released_at."""
    adapter = FREDAdapter(
        fred_id,
        cache_dir=cache_dir,
        refresh=refresh,
    )

    return _canonical_three_col(adapter.fetch())


# ---------------------------------------------------------------------------
# USD/CAD target construction
# ---------------------------------------------------------------------------


def _build_forward_log_return_frame(
    price_df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Build forward cumulative log-return observations.

    For timestamp t:

        value[t] = log(price[t + window] / price[t])

    This is deliberately a FORWARD return because the forecasting task
    should predict the return occurring after the forecast origin.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")

    required_columns = {"timestamp", "value"}

    missing = required_columns - set(price_df.columns)

    if missing:
        raise RuntimeError(
            f"USD/CAD price data missing required columns: {sorted(missing)}"
        )

    frame = (
        price_df[["timestamp", "value"]]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    frame = frame[
        frame["value"].notna()
        & (frame["value"] > 0)
    ].reset_index(drop=True)

    # Forward cumulative log return:
    #
    # log(P[t+h] / P[t])
    #
    frame["value"] = np.log(
        frame["value"].shift(-window) / frame["value"]
    )

    frame = frame.dropna(subset=["value"]).reset_index(drop=True)

    # The forward return becomes known when the future exchange rate
    # is observed, but the target itself is associated with the
    # original forecast timestamp.
    #
    # DataService uses the forecast timestamp to resolve the target.
    frame["released_at"] = (
        pd.to_datetime(frame["timestamp"])
        + pd.offsets.BDay(window)
    )

    return frame[["timestamp", "value", "released_at"]]


def build_usdcad_log_return_service(
    *,
    windows: tuple[int, ...] = USDCAD_RETURN_WINDOWS,
    refresh: bool = False,
    start: str = "1990-01-01",
    end: str | None = None,
    fred_cache_dir: Path | None = None,
) -> DataService:
    """Build a DataService containing USD/CAD return targets."""

    _load_fred_dotenv()

    cache_dir = fred_cache_dir or DEFAULT_FRED_CACHE_DIR

    cache_dir = cache_dir.resolve()

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    usdcad = _fred_frame(
        USDCAD_SERIES_ID,
        cache_dir=cache_dir,
        refresh=refresh,
    )

    usdcad["timestamp"] = pd.to_datetime(usdcad["timestamp"])

    if start is not None:
        usdcad = usdcad[
            usdcad["timestamp"] >= pd.Timestamp(start)
        ]

    if end is not None:
        usdcad = usdcad[
            usdcad["timestamp"] < pd.Timestamp(end)
        ]

    usdcad = usdcad.reset_index(drop=True)

    if usdcad.empty:
        raise RuntimeError(
            "No USD/CAD observations available after applying "
            f"start={start!r}, end={end!r}."
        )

    service = DataService()

    for window in windows:
        series_id = usdcad_logret_series_id(window)

        label = USDCAD_WINDOW_LABELS.get(
            window,
            f"{window} business days",
        )

        target_frame = _build_forward_log_return_frame(
            usdcad,
            window,
        )

        service.register(
            series_id,
            StaticFrameAdapter(target_frame),
            SeriesMetadata(
                series_id=series_id,
                description=(
                    "USD/CAD forward cumulative log return over "
                    f"{window} business day(s) ({label}), "
                    "derived from FRED DEXCAUS."
                ),
                source="FRED (DEXCAUS), derived",
                units="log-return",
                frequency="B",
                table_id=f"fred:DEXCAUS:logret-{window}b",
            ),
        )

    return service


# ---------------------------------------------------------------------------
# Daily FRED features
# ---------------------------------------------------------------------------


def _build_daily_fred_level_feature(
    fred_id: str,
    *,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    """Build a lagged daily FRED level feature."""

    frame = _fred_frame(
        fred_id,
        cache_dir=cache_dir,
        refresh=refresh,
    )

    frame = _drop_weekend_timestamp_rows(frame)

    frame = _business_daily_ffill(frame)

    return _apply_one_business_day_feature_lag(frame)


def _build_daily_fred_return_feature(
    fred_id: str,
    *,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    """Build a lagged daily FRED log-return feature."""

    frame = _fred_frame(
        fred_id,
        cache_dir=cache_dir,
        refresh=refresh,
    )

    frame = _drop_weekend_timestamp_rows(frame)

    frame["value"] = pd.to_numeric(
        frame["value"],
        errors="coerce",
    )

    frame = frame[
        frame["value"].notna()
        & (frame["value"] > 0)
    ].reset_index(drop=True)

    # Fill missing business days before calculating returns.
    frame = _business_daily_ffill(frame)

    frame["value"] = np.log(
        frame["value"]
        / frame["value"].shift(1)
    )

    frame = frame.dropna(
        subset=["value"]
    ).reset_index(drop=True)

    return _apply_one_business_day_feature_lag(frame)


# ---------------------------------------------------------------------------
# Monthly macroeconomic features
# ---------------------------------------------------------------------------


def _build_monthly_cpi_feature(
    *,
    cache_dir: Path,
    refresh: bool,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Build a conservative, lagged monthly CPI growth feature."""

    cpi = _fred_frame(
        "CPIAUCSL",
        cache_dir=cache_dir,
        refresh=refresh,
    )

    cpi["value"] = pd.to_numeric(
        cpi["value"],
        errors="coerce",
    )

    cpi = cpi[
        cpi["value"].notna()
        & (cpi["value"] > 0)
    ].reset_index(drop=True)

    # Month-over-month log change.
    cpi["value"] = np.log(
        cpi["value"]
        / cpi["value"].shift(1)
    )

    cpi = cpi.dropna(
        subset=["value"]
    ).reset_index(drop=True)

    # Conservative release-date proxy.
    #
    # We deliberately do not assume the observation is available
    # on the reference month's timestamp.
    cpi["released_at"] = (
        pd.to_datetime(cpi["timestamp"])
        + pd.offsets.MonthEnd(1)
        + pd.offsets.BDay(10)
    )

    daily = _business_daily_expand_from_releases(
        cpi,
        start=start,
        end=end,
    )

    return _apply_one_business_day_feature_lag(daily)


def _build_monthly_unemployment_feature(
    *,
    cache_dir: Path,
    refresh: bool,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Build a conservative, lagged monthly unemployment feature."""

    unemployment = _fred_frame(
        "UNRATE",
        cache_dir=cache_dir,
        refresh=refresh,
    )

    unemployment["value"] = pd.to_numeric(
        unemployment["value"],
        errors="coerce",
    )

    unemployment = unemployment.dropna(
        subset=["value"]
    ).reset_index(drop=True)

    # Conservative publication-date proxy.
    unemployment["released_at"] = (
        pd.to_datetime(unemployment["timestamp"])
        + pd.offsets.MonthEnd(1)
        + pd.offsets.BDay(10)
    )

    daily = _business_daily_expand_from_releases(
        unemployment,
        start=start,
        end=end,
    )

    return _apply_one_business_day_feature_lag(daily)


# ---------------------------------------------------------------------------
# Main multivariate service
# ---------------------------------------------------------------------------


def build_usdcad_multivariate_service(  # noqa: PLR0912, PLR0915
    *,
    windows: tuple[int, ...] = USDCAD_RETURN_WINDOWS,
    include_covariates: bool = True,
    covariate_series_ids: list[str] | None = None,
    strict_covariates: bool = False,
    refresh: bool = False,
    start: str = "1990-01-01",
    end: str | None = None,
    fred_cache_dir: Path | None = None,
) -> DataService:
    """Build a leak-safe USD/CAD forecasting DataService.

    Parameters
    ----------
    windows:
        Forecast horizons in business days.

    include_covariates:
        Whether to register macroeconomic/FRED covariates.

    covariate_series_ids:
        Optional list selecting which covariates to include.

    strict_covariates:
        If True, a covariate failure raises immediately.
        If False, failed covariates are skipped with a warning.

    refresh:
        Force FREDAdapter to refresh cached data.

    start:
        Start date for the dataset.

    end:
        Optional exclusive end date.
    """

    _load_fred_dotenv()

    service = build_usdcad_log_return_service(
        windows=windows,
        refresh=refresh,
        start=start,
        end=end,
        fred_cache_dir=fred_cache_dir,
    )

    if not include_covariates:
        return service

    cache_dir = fred_cache_dir or DEFAULT_FRED_CACHE_DIR

    cache_dir = cache_dir.resolve()

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    desired = (
        set(covariate_series_ids)
        if covariate_series_ids is not None
        else set(DEFAULT_COVARIATE_SERIES_IDS)
    )

    def _handle_error(
        series_id: str,
        exc: Exception,
    ) -> None:
        if strict_covariates:
            raise RuntimeError(
                f"Failed to build required covariate "
                f"{series_id!r}."
            ) from exc

        warnings.warn(
            f"Skipping unavailable covariate "
            f"{series_id!r}: {exc}",
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # USD/CAD lagged return
    # ------------------------------------------------------------------

    if SERIES_ID_USDCAD_RETURN in desired:
        try:
            usdcad_return = _build_daily_fred_return_feature(
                USDCAD_SERIES_ID,
                cache_dir=cache_dir,
                refresh=refresh,
            )

            service.register(
                SERIES_ID_USDCAD_RETURN,
                StaticFrameAdapter(usdcad_return),
                SeriesMetadata(
                    series_id=SERIES_ID_USDCAD_RETURN,
                    description=(
                        "USD/CAD daily log return, "
                        "lagged one business day."
                    ),
                    source="FRED (DEXCAUS), derived",
                    units="log-return",
                    frequency="B",
                    table_id="fred:DEXCAUS:log-return-l1b",
                ),
            )

        except (RuntimeError, ValueError) as exc:
            _handle_error(
                SERIES_ID_USDCAD_RETURN,
                exc,
            )

    # ------------------------------------------------------------------
    # Federal Funds Rate
    # ------------------------------------------------------------------

    if SERIES_ID_FED_FUNDS in desired:
        try:
            fed_funds = _build_daily_fred_level_feature(
                "DFF",
                cache_dir=cache_dir,
                refresh=refresh,
            )

            service.register(
                SERIES_ID_FED_FUNDS,
                StaticFrameAdapter(fed_funds),
                SeriesMetadata(
                    series_id=SERIES_ID_FED_FUNDS,
                    description=(
                        "US effective federal funds rate, "
                        "lagged one business day."
                    ),
                    source="FRED (DFF)",
                    units="percent",
                    frequency="B",
                    table_id="fred:DFF:l1b",
                ),
            )

        except (RuntimeError, ValueError) as exc:
            _handle_error(
                SERIES_ID_FED_FUNDS,
                exc,
            )

    # ------------------------------------------------------------------
    # US 2-Year Treasury
    # ------------------------------------------------------------------

    if SERIES_ID_US_2Y in desired:
        try:
            us_2y = _build_daily_fred_level_feature(
                "DGS2",
                cache_dir=cache_dir,
                refresh=refresh,
            )

            service.register(
                SERIES_ID_US_2Y,
                StaticFrameAdapter(us_2y),
                SeriesMetadata(
                    series_id=SERIES_ID_US_2Y,
                    description=(
                        "US 2-year Treasury yield, "
                        "lagged one business day."
                    ),
                    source="FRED (DGS2)",
                    units="percent",
                    frequency="B",
                    table_id="fred:DGS2:l1b",
                ),
            )

        except (RuntimeError, ValueError) as exc:
            _handle_error(
                SERIES_ID_US_2Y,
                exc,
            )

    # ------------------------------------------------------------------
    # US 10-Year Treasury
    # ------------------------------------------------------------------

    if SERIES_ID_US_10Y in desired:
        try:
            us_10y = _build_daily_fred_level_feature(
                "DGS10",
                cache_dir=cache_dir,
                refresh=refresh,
            )

            service.register(
                SERIES_ID_US_10Y,
                StaticFrameAdapter(us_10y),
                SeriesMetadata(
                    series_id=SERIES_ID_US_10Y,
                    description=(
                        "US 10-year Treasury yield, "
                        "lagged one business day."
                    ),
                    source="FRED (DGS10)",
                    units="percent",
                    frequency="B",
                    table_id="fred:DGS10:l1b",
                ),
            )

        except (RuntimeError, ValueError) as exc:
            _handle_error(
                SERIES_ID_US_10Y,
                exc,
            )

    # ------------------------------------------------------------------
    # US 2Y-10Y yield spread
    # ------------------------------------------------------------------

    if SERIES_ID_US_2Y10Y_SPREAD in desired:
        try:
            us_2y = _fred_frame(
                "DGS2",
                cache_dir=cache_dir,
                refresh=refresh,
            )

            us_10y = _fred_frame(
                "DGS10",
                cache_dir=cache_dir,
                refresh=refresh,
            )

            spread = pd.merge(
                us_10y[["timestamp", "value"]],
                us_2y[["timestamp", "value"]],
                on="timestamp",
                how="inner",
                suffixes=("_10y", "_2y"),
            )

            spread["value"] = (
                spread["value_10y"]
                - spread["value_2y"]
            )

            spread["released_at"] = pd.to_datetime(
                spread["timestamp"]
            ) + pd.offsets.BDay(1)

            spread = spread[
                ["timestamp", "value", "released_at"]
            ]

            spread = _drop_weekend_timestamp_rows(
                spread
            )

            spread = _business_daily_ffill(
                spread
            )

            spread = _apply_one_business_day_feature_lag(
                spread
            )

            service.register(
                SERIES_ID_US_2Y10Y_SPREAD,
                StaticFrameAdapter(spread),
                SeriesMetadata(
                    series_id=SERIES_ID_US_2Y10Y_SPREAD,
                    description=(
                        "US 10-year Treasury yield minus "
                        "US 2-year Treasury yield, "
                        "lagged one business day."
                    ),
                    source="FRED (DGS10 - DGS2), derived",
                    units="percentage-points",
                    frequency="B",
                    table_id="fred:DGS10-DGS2:spread-l1b",
                ),
            )

        except (RuntimeError, ValueError) as exc:
            _handle_error(
                SERIES_ID_US_2Y10Y_SPREAD,
                exc,
            )

    # ------------------------------------------------------------------
    # CPI
    # ------------------------------------------------------------------

    if SERIES_ID_CPI in desired:
        try:
            cpi = _build_monthly_cpi_feature(
                cache_dir=cache_dir,
                refresh=refresh,
                start=start,
                end=end,
            )

            service.register(
                SERIES_ID_CPI,
                StaticFrameAdapter(cpi),
                SeriesMetadata(
                    series_id=SERIES_ID_CPI,
                    description=(
                        "US CPI month-over-month log change, "
                        "expanded from a conservative release date "
                        "and lagged one business day."
                    ),
                    source="FRED (CPIAUCSL), derived",
                    units="log-change",
                    frequency="B",
                    table_id="fred:CPIAUCSL:mom-logdiff-l1b",
                ),
            )

        except (RuntimeError, ValueError) as exc:
            _handle_error(
                SERIES_ID_CPI,
                exc,
            )

    # ------------------------------------------------------------------
    # Unemployment
    # ------------------------------------------------------------------

    if SERIES_ID_UNEMPLOYMENT in desired:
        try:
            unemployment = _build_monthly_unemployment_feature(
                cache_dir=cache_dir,
                refresh=refresh,
                start=start,
                end=end,
            )

            service.register(
                SERIES_ID_UNEMPLOYMENT,
                StaticFrameAdapter(unemployment),
                SeriesMetadata(
                    series_id=SERIES_ID_UNEMPLOYMENT,
                    description=(
                        "US unemployment rate, "
                        "expanded from a conservative release date "
                        "and lagged one business day."
                    ),
                    source="FRED (UNRATE)",
                    units="percent",
                    frequency="B",
                    table_id="fred:UNRATE:l1b",
                ),
            )

        except (RuntimeError, ValueError) as exc:
            _handle_error(
                SERIES_ID_UNEMPLOYMENT,
                exc,
            )

    # ------------------------------------------------------------------
    # Oil return
    # ------------------------------------------------------------------

    if SERIES_ID_OIL_RETURN in desired:
        try:
            oil_return = _build_daily_fred_return_feature(
                "DCOILWTICO",
                cache_dir=cache_dir,
                refresh=refresh,
            )

            service.register(
                SERIES_ID_OIL_RETURN,
                StaticFrameAdapter(oil_return),
                SeriesMetadata(
                    series_id=SERIES_ID_OIL_RETURN,
                    description=(
                        "WTI crude oil daily log return, "
                        "lagged one business day."
                    ),
                    source="FRED (DCOILWTICO), derived",
                    units="log-return",
                    frequency="B",
                    table_id="fred:DCOILWTICO:log-return-l1b",
                ),
            )

        except (RuntimeError, ValueError) as exc:
            _handle_error(
                SERIES_ID_OIL_RETURN,
                exc,
            )

    return service