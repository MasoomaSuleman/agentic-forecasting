"""Leak-safe CAD/USD exchange-rate data service.

The first POC slice registers CAD/USD cumulative log-return targets from FRED
``DEXCAUS`` and an optional WTI return covariate from ``DCOILWTICO``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.fred import FREDAdapter
from aieng.forecasting.data.features import (
    StaticFrameAdapter,
    apply_one_business_day_feature_lag,
    to_log_return_feature,
)
from dotenv import load_dotenv


CADUSD_FRED_SERIES_ID = "DEXCAUS"
WTI_FRED_SERIES_ID = "DCOILWTICO"
CADUSD_SERIES_ID = "cadusd_exchange_rate"
CADUSD_RETURN_WINDOWS: tuple[int, ...] = (1, 5, 21)
DEFAULT_CADUSD_COVARIATES = ["wti_log_ret_1b_l1b"]


def _repo_root() -> Path:
    """Find the repository root for dotenv loading and cache paths."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "aieng-forecasting").is_dir() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not find the agentic-forecasting repository root.")


def _load_fred_api_key() -> str:
    """Load the FRED credential from the repository's ignored .env file."""
    root = _repo_root()
    load_dotenv(root / ".env", override=False)
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(f"FRED_API_KEY is missing. Add it to {root / '.env'}.")
    return api_key


def _apply_date_range(frame: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    """Restrict fetched observations to the requested date range."""
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out[out["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["timestamp"] < pd.Timestamp(end)]
    if out.empty:
        raise RuntimeError(f"No FRED observations remain for start={start!r}, end={end!r}.")
    return out.reset_index(drop=True)


def _log_return_frame(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Build a positive-price cumulative log-return series over ``window`` days."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    out = frame[["timestamp", "value"]].copy().sort_values("timestamp")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    out = out[out["value"] > 0].copy()
    out["value"] = np.log(out["value"] / out["value"].shift(window))
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    out["released_at"] = pd.to_datetime(out["timestamp"])
    return out[["timestamp", "value", "released_at"]]


def _register_return_target(service: DataService, rate_frame: pd.DataFrame, window: int) -> str:
    """Register one CAD/USD return target and return its series id."""
    series_id = f"cadusd_logret_{window}b"
    service.register(
        series_id,
        StaticFrameAdapter(_log_return_frame(rate_frame, window)),
        SeriesMetadata(
            series_id=series_id,
            description=f"CAD/USD cumulative log return over {window} business day(s), derived from FRED DEXCAUS",
            source=f"FRED ({CADUSD_FRED_SERIES_ID}), derived",
            units="log-return",
            frequency="B",
            table_id=f"fred:{CADUSD_FRED_SERIES_ID}:logret-{window}b",
        ),
    )
    return series_id


def build_cadusd_service(
    *,
    windows: tuple[int, ...] = CADUSD_RETURN_WINDOWS,
    include_covariates: bool = True,
    covariate_series_ids: list[str] | None = None,
    start: str = "1990-01-01",
    end: str | None = None,
    api_key: str | None = None,
) -> DataService:
    """Build a CAD/USD ``DataService`` with targets and optional WTI returns."""
    fred_api_key = api_key or _load_fred_api_key()
    service = DataService()

    rate_frame = FREDAdapter(
        series_id=CADUSD_FRED_SERIES_ID,
        api_key=fred_api_key,
    ).fetch()
    rate_frame = _apply_date_range(rate_frame, start, end)
    for window in windows:
        _register_return_target(service, rate_frame, window)

    if not include_covariates:
        return service

    requested = set(covariate_series_ids or DEFAULT_CADUSD_COVARIATES)
    if "wti_log_ret_1b_l1b" in requested:
        oil_frame = FREDAdapter(
            series_id=WTI_FRED_SERIES_ID,
            api_key=fred_api_key,
        ).fetch()
        oil_frame = _apply_date_range(oil_frame, start, end)
        wti_returns = apply_one_business_day_feature_lag(to_log_return_feature(oil_frame))
        service.register(
            "wti_log_ret_1b_l1b",
            StaticFrameAdapter(wti_returns),
            SeriesMetadata(
                series_id="wti_log_ret_1b_l1b",
                description="WTI daily log return, lagged one business day",
                source=f"FRED ({WTI_FRED_SERIES_ID}), derived",
                units="log-return",
                frequency="B",
                table_id=f"fred:{WTI_FRED_SERIES_ID}:logret-1b-l1b",
            ),
        )

    return service


__all__ = [
    "CADUSD_FRED_SERIES_ID",
    "CADUSD_RETURN_WINDOWS",
    "CADUSD_SERIES_ID",
    "DEFAULT_CADUSD_COVARIATES",
    "WTI_FRED_SERIES_ID",
    "build_cadusd_service",
]
