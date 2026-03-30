"""Scenario runtime — thin Python wrapper around the Rust native backend.

[`Scenario`][tradingflow.Scenario] specifies a directed acyclic graph of
sources and operators. The POCQ event loop and DAG propagation are
implemented in Rust (`tradingflow._native`).
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from .operator import Operator, NativeOperator
from .source import Source, NativeSource
from .types import Handle

from tradingflow._native import NativeScenario


class Scenario:
    """A directed acyclic graph of sources and operators.

    Sources and operators are registered via `add_source` and
    `add_operator`, each returning a `Handle`. To record history,
    use the `record` operator explicitly. `run` executes the POCQ
    event loop.
    """

    __slots__ = ("_native", "_sources")

    def __init__(self) -> None:
        self._native = NativeScenario()
        self._sources: list[tuple[Source, Any, Any]] = []

    def add_source(self, source: Source | NativeSource) -> Handle:
        """Register a source and return a handle to its output node."""
        if isinstance(source, NativeSource):
            idx = self._native.add_native_source(
                source.kind,
                source.dtype,
                list(source.shape),
                source.params,
            )
            return Handle(idx, source.shape, np.dtype(source.dtype))
        else:
            idx, hist_sender, live_sender = self._native.add_py_source(
                list(source.shape),
                str(source.dtype),
            )
            self._sources.append((source, hist_sender, live_sender))
            return Handle(idx, source.shape, source.dtype)

    def add_operator(
        self,
        operator: NativeOperator | Operator,
        *,
        clock: Handle | None = None,
    ) -> Handle:
        """Register an operator and return a handle to its output node.

        Parameters
        ----------
        operator
            The operator to register (native or Python).
        clock
            Optional clock handle. If provided, the operator is triggered
            by the clock instead of its inputs. The clock is not an input —
            the operator does not read its value.
        """
        input_indices = [inp.index for inp in operator.inputs]
        if isinstance(operator, NativeOperator):
            idx = self._native.add_native_operator(
                operator.kind,
                str(operator.dtype),
                input_indices,
                list(operator.shape),
                operator.params,
                clock_index=clock.index if clock else None,
            )
        else:
            input_names, output_name = operator.get_io_types()
            idx = self._native.add_py_operator(
                input_indices,
                input_names,
                output_name,
                list(operator.shape),
                operator,
                operator.init_state(),
                clock_index=clock.index if clock else None,
            )
        return Handle(idx, operator.shape, operator.dtype)

    # -- Series access --------------------------------------------------------

    def series_len(self, handle: Handle) -> int:
        """Number of recorded elements in a Series node."""
        return self._native.series_len(handle.index)

    def series_timestamps(self, handle: Handle) -> np.ndarray:
        """Recorded timestamps as datetime64[ns] array."""
        ts_i64 = np.asarray(self._native.series_timestamps(handle.index))
        return ts_i64.view("datetime64[ns]")

    def series_values(self, handle: Handle) -> np.ndarray:
        """Recorded values as numpy array."""
        vals_raw = np.asarray(self._native.series_values(handle.index))
        n = self._native.series_len(handle.index)
        if handle.shape:
            return vals_raw.reshape(n, *handle.shape)
        return vals_raw

    # -- Execution ------------------------------------------------------------

    def run(self) -> None:
        """Execute the POCQ event loop.

        Python sources are driven concurrently on a background thread.
        The Rust runtime runs on the main thread.
        """
        driver_error: list[BaseException | None] = [None]

        async def _drive_one(
            source: Source,
            hist_sender: Any,
            live_sender: Any,
        ) -> None:
            try:
                hist_iter, live_iter = source.subscribe()
                last_ts: np.datetime64 | None = None
                async for raw_ts, raw_val in hist_iter:
                    ts = _coerce_timestamp(raw_ts)
                    if last_ts is not None and ts < last_ts:
                        raise ValueError(
                            f"Source '{source.name}' emitted timestamp {ts!r} which is "
                            f"less than last committed timestamp {last_ts!r}."
                        )
                    last_ts = ts
                    val = np.ascontiguousarray(np.asarray(raw_val, dtype=source.dtype))
                    ts_ns = int(ts.view("int64"))
                    hist_sender.send(ts_ns, val)
                hist_sender.close()
                async for raw_val in live_iter:
                    val = np.ascontiguousarray(np.asarray(raw_val, dtype=source.dtype))
                    live_sender.send(val)
                live_sender.close()
            except Exception:
                hist_sender.close()
                live_sender.close()
                raise

        async def _drive_all() -> None:
            tasks = [asyncio.create_task(_drive_one(src, hs, ls)) for src, hs, ls in self._sources]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for _, hs, ls in self._sources:
                    hs.close()
                    ls.close()
                raise

        def driver() -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_drive_all())
            except BaseException as exc:
                driver_error[0] = exc
            finally:
                loop.close()

        if self._sources:
            self._native.run_with_driver(driver)
        else:
            self._native.run()

        if driver_error[0] is not None:
            raise driver_error[0]


def _coerce_timestamp(value: np.datetime64) -> np.datetime64:
    try:
        timestamp = np.datetime64(value)
    except Exception as exc:
        raise ValueError(f"Could not parse timestamp value {value!r}.") from exc
    return timestamp.astype("datetime64[ns]")
