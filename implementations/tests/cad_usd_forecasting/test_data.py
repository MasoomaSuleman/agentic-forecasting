"""Tests for the first CAD/USD data pipeline slice."""

import numpy as np
import pandas as pd

from implementations.cad_usd_forecasting.data import _apply_date_range, _log_return_frame


def test_log_return_frame_builds_expected_cumulative_returns() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="B"),
            "value": [1.0, 1.1, 1.21],
        }
    )

    result = _log_return_frame(frame, window=1)

    np.testing.assert_allclose(result["value"].to_numpy(), [np.log(1.1), np.log(1.1)], rtol=1e-12)


def test_date_range_filters_observations() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="B"),
            "value": [1.0, 1.1, 1.21],
        }
    )

    result = _apply_date_range(frame, "2026-01-02", "2026-01-06")

    assert result["timestamp"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-05"]
