"""Tests for source adapters."""

from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import numpy as np
import pytest

from src import ArrayBundleSource, AsyncCallableSource, CSVSource, Scenario, Series


def ts(i: int) -> np.datetime64:
    """Create a nanosecond timestamp from an integer."""
    return np.datetime64(i, "ns")


class TestCSVSource:
    def test_csv_source_with_custom_columns_and_converters(self, tmp_path: Path) -> None:
        path = tmp_path / "prices.csv"
        path.write_text(
            "id,t_ns,price_text\n"
            "a,1,10.5\n"
            "b,2,20.25\n",
            encoding="utf-8",
        )

        series = Series((), np.dtype(np.float32))
        source = CSVSource(
            path=path,
            series=series,
            timestamp_col="t_ns",
            value_cols=("price_text",),
            timestamp_parser=lambda raw: np.datetime64(int(raw), "ns"),
            converters={"price_text": float},
        )
        scenario = Scenario(sources=(source,))
        asyncio.run(scenario.run())

        assert list(series.index) == [ts(1), ts(2)]
        assert series.values.dtype == np.dtype(np.float32)
        assert list(series.values) == pytest.approx([10.5, 20.25])

    def test_csv_source_rejects_shape_mismatch_configuration(self, tmp_path: Path) -> None:
        path = tmp_path / "vectors.csv"
        path.write_text("t,a\n1,1.0\n", encoding="utf-8")

        series = Series((2,), np.dtype(np.float64))
        with pytest.raises(ValueError, match="value_cols count"):
            CSVSource(
                path=path,
                series=series,
                timestamp_col="t",
                value_cols=("a",),
            )

    def test_csv_source_rejects_missing_required_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "missing_cols.csv"
        path.write_text("t,x\n1,1.0\n", encoding="utf-8")

        series = Series((), np.dtype(np.float64))
        source = CSVSource(
            path=path,
            series=series,
            timestamp_col="t",
            value_cols=("price",),
        )
        scenario = Scenario(sources=(source,))

        with pytest.raises(ValueError, match="missing required columns"):
            asyncio.run(scenario.run())


class TestArrayBundleSource:
    def test_from_arrays(self) -> None:
        series = Series((), np.dtype(np.float64))
        source = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([1.0, 2.0]),
            series=series,
        )

        scenario = Scenario(sources=(source,))
        asyncio.run(scenario.run())
        assert list(series.index) == [ts(1), ts(2)]
        assert list(series.values) == pytest.approx([1.0, 2.0])

    def test_from_pickle(self, tmp_path: Path) -> None:
        path = tmp_path / "bundle.pkl"
        payload = {
            "timestamps": np.array([ts(3), ts(4)]),
            "values": np.array([3.0, 4.0]),
        }
        path.write_bytes(pickle.dumps(payload))

        series = Series((), np.dtype(np.float64))
        source = ArrayBundleSource.from_pickle(path, series)
        scenario = Scenario(sources=(source,))
        asyncio.run(scenario.run())

        assert list(series.index) == [ts(3), ts(4)]
        assert list(series.values) == pytest.approx([3.0, 4.0])

    def test_rejects_length_mismatch(self) -> None:
        series = Series((), np.dtype(np.float64))
        with pytest.raises(ValueError, match="length"):
            ArrayBundleSource.from_arrays(
                timestamps=np.array([ts(1), ts(2)]),
                values=np.array([1.0]),
                series=series,
            )


class TestAsyncCallableSource:
    def test_ingest_timestamps_are_strictly_increasing(self) -> None:
        series = Series((), np.dtype(np.float64))

        async def stream():
            # Intentionally no sleeps; timestamps may collide and must be normalized.
            yield 1.0
            yield 2.0
            yield 3.0

        source = AsyncCallableSource(series, stream)
        scenario = Scenario(sources=(source,))
        asyncio.run(scenario.run())

        assert len(series) == 3
        diffs = np.diff(series.index.astype("int64"))
        assert np.all(diffs > 0)
