"""Tests for daily market snapshot source adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from src import Scenario
from src.ops import select
from src.sources.eastmoney.history import DailyMarketSnapshotCSVSource


def dt(text: str) -> np.datetime64:
    """Creates datetime64[ns] timestamp."""
    return np.datetime64(text).astype("datetime64[ns]")


class TestDailyMarketSnapshotCSVSource:
    def test_parses_raw_daily_price_and_scales_volume(self, tmp_path: Path) -> None:
        path = tmp_path / "daily.csv"
        path.write_text(
            "date,open,close,high,low,volume,amount\n" "2022-01-03,10.0,11.0,12.0,9.0,5,1000.0\n",
            encoding="utf-8",
        )

        source = DailyMarketSnapshotCSVSource(path)
        scenario = Scenario(sources=(source,))
        asyncio.run(scenario.run())

        assert len(source.series) == 1
        assert source.series.index[0] == dt("2022-01-03")
        assert list(source.schema.field_ids) == ["open", "close", "high", "low", "amount", "volume"]
        np.testing.assert_array_almost_equal(
            source.series.values[0],
            np.array([10.0, 11.0, 12.0, 9.0, 1000.0, 500.0], dtype=np.float64),
        )

    def test_rejects_invalid_ohlc_row_in_strict_mode(self, tmp_path: Path) -> None:
        path = tmp_path / "invalid_daily.csv"
        path.write_text(
            "date,open,close,high,low,volume,amount\n" "2022-01-03,10.0,11.0,10.5,9.0,5,1000.0\n",
            encoding="utf-8",
        )

        source = DailyMarketSnapshotCSVSource(path, strict_row_checks=True)
        scenario = Scenario(sources=(source,))
        with pytest.raises(ValueError, match="sanity checks"):
            asyncio.run(scenario.run())

    def test_drops_invalid_rows_when_not_strict(self, tmp_path: Path) -> None:
        path = tmp_path / "mixed_daily.csv"
        path.write_text(
            "date,open,close,high,low,volume,amount\n"
            "2022-01-03,10.0,11.0,10.5,9.0,5,1000.0\n"
            "2022-01-04,11.0,10.0,12.0,9.5,6,1200.0\n",
            encoding="utf-8",
        )

        source = DailyMarketSnapshotCSVSource(path, strict_row_checks=False)
        scenario = Scenario(sources=(source,))
        asyncio.run(scenario.run())

        assert len(source.series) == 1
        assert source.diagnostics.dropped_rows == 1
        assert source.diagnostics.total_rows == 2
        assert source.diagnostics.emitted_rows == 1
        assert source.series.index[0] == dt("2022-01-04")

    def test_missing_required_columns_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "missing_cols.csv"
        path.write_text("date,open,close,high,low,volume\n2022-01-03,10,11,12,9,5\n", encoding="utf-8")

        source = DailyMarketSnapshotCSVSource(path)
        scenario = Scenario(sources=(source,))
        with pytest.raises(ValueError, match="missing required columns"):
            asyncio.run(scenario.run())

    def test_select_fields_integration(self, tmp_path: Path) -> None:
        path = tmp_path / "daily.csv"
        path.write_text(
            "date,open,close,high,low,volume,amount\n"
            "2022-01-03,10.0,11.0,12.0,9.0,5,1000.0\n"
            "2022-01-04,11.0,13.0,14.0,10.0,6,1500.0\n",
            encoding="utf-8",
        )

        source = DailyMarketSnapshotCSVSource(path)
        close_index = source.schema.field_index["close"]
        close_selector = select(source.series, (close_index,))

        scenario = Scenario(sources=(source,), operators=(close_selector,))
        asyncio.run(scenario.run())

        assert len(close_selector.output) == 2
        np.testing.assert_array_almost_equal(
            close_selector.output.values,
            np.array([[11.0], [13.0]], dtype=np.float64),
        )
