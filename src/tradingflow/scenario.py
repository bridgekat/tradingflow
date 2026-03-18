"""Scenario runtime — thin Python wrapper around the Rust native backend.

[`Scenario`][tradingflow.Scenario] specifies a directed acyclic graph of
sources and operators.  The POCQ event loop and DAG propagation are
implemented in Rust (``tradingflow_native``).  Python sources are driven
concurrently on a background asyncio thread; events flow through channels
to the Rust runtime on the main thread.

Point-of-coherency queue (POCQ)
-------------------------------
Each source exposes a `(historical, live)` iterator pair via
[`Source.subscribe`][tradingflow.Source.subscribe].

* **Historical constraint** – before advancing the POCQ, every active
  historical iterator must have a pending event ready.
* **Live iterators** are exempt from this constraint.
* The POCQ accumulates events sharing the same timestamp.  When an event
  with a strictly larger timestamp arrives, all queued events are flushed.
* After all iterators are exhausted, any remaining queued events are flushed.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import numpy as np

from .observable import Observable
from .operator import NativeOperator, Operator
from .series import Series
from .source import Source

from tradingflow_native import NativeScenario

type _AnyObservable = Observable[Any, Any]
type _AnySeries = Series[Any, Any]
type _AnySource = Source[Any, Any]
type _AnyOperator = Operator[Any, Any, Any, Any]


class Scenario:
    """A directed acyclic graph of sources and operators.

    Sources and operators are registered via [`add_source`][.add_source] and
    [`add_operator`][.add_operator], each returning the output
    [`Observable`][tradingflow.Observable] that will be updated during
    [`run`][.run].  Observables can be *materialized* into
    [`Series`][tradingflow.Series] via [`materialize`][.materialize].
    """

    __slots__ = (
        "_sources",
        "_operators",
        "_materializations",
    )

    _sources: list[tuple[_AnySource, _AnyObservable]]
    _operators: list[tuple[_AnyOperator, _AnyObservable]]
    _materializations: dict[int, _AnySeries]  # id(observable) -> series

    def __init__(self) -> None:
        self._sources = []
        self._operators = []
        self._materializations = {}

    @property
    def sources(self) -> list[tuple[_AnySource, _AnyObservable]]:
        """Registered sources in insertion order."""
        return self._sources

    @property
    def operators(self) -> list[tuple[_AnyOperator, _AnyObservable]]:
        """Registered operators in insertion order."""
        return self._operators

    def add_source(self, source: _AnySource) -> _AnyObservable:
        """Register a source and return its output observable."""
        observable = Observable(source.shape, source.dtype)
        self._sources.append((source, observable))
        return observable

    def add_operator(self, operator: _AnyOperator) -> _AnyObservable:
        """Register an operator and return its output observable."""
        observable = Observable(operator.shape, operator.dtype)
        self._operators.append((operator, observable))
        return observable

    def materialize(self, observable: _AnyObservable) -> _AnySeries:
        """Materialize an observable: allocate a series to store full history.

        Returns the [`Series`][tradingflow.Series] that will be populated
        during [`run`][.run].
        """
        key = id(observable)
        if key in self._materializations:
            return self._materializations[key]
        series = Series(observable.shape, observable.dtype)
        self._materializations[key] = series
        return series

    async def run(self) -> None:
        """Consume all source streams and propagate to operators.

        Delegates to the Rust POCQ runtime.  Python sources are driven
        concurrently on a background thread; Rust and Python operators
        are dispatched during the DAG flush.
        """
        native = NativeScenario()
        obs_to_idx: dict[int, int] = {}

        def _resolve_obs_id(inp: Any) -> int:
            if isinstance(inp, Observable):
                return id(inp)
            if isinstance(inp, Series):
                for obs_id, s in self._materializations.items():
                    if s is inp:
                        return obs_id
                raise ValueError("Series input must be a materialized series from this scenario.")
            raise TypeError(f"Unsupported input type: {type(inp)}")

        # -- 1. Register sources (channel-based) -------------------------------
        source_infos: list[tuple[Source[Any, Any], Observable[Any, Any], object, object]] = []
        for source, obs in self._sources:
            dtype_str = str(source.dtype)
            initial_flat = np.ascontiguousarray(source.initial).ravel()
            idx, hist_sender, live_sender = native.add_channel_source(
                list(obs.shape),
                dtype_str,
                initial_flat,
            )
            obs_to_idx[id(obs)] = idx
            source_infos.append((source, obs, hist_sender, live_sender))

        # -- 2. Register operators ---------------------------------------------
        for operator, obs in self._operators:
            input_indices: list[int] = []
            for inp in operator.inputs:
                inp_id = _resolve_obs_id(inp)
                node_idx = obs_to_idx[inp_id]
                input_indices.append(node_idx)

            dtype_str = str(operator.dtype)

            if isinstance(operator, NativeOperator):
                # Route to Rust-native operator via opaque handle.
                handle = operator.create_handle()
                idx = native.register_handle_operator(
                    handle,
                    input_indices,
                    list(obs.shape),
                )
            else:
                # Python operator — uses GIL callback during flush.
                input_views: list[object] = []
                for inp in operator.inputs:
                    inp_id = _resolve_obs_id(inp)
                    node_idx = obs_to_idx[inp_id]
                    if isinstance(inp, Series):
                        input_views.append(native.series_view(node_idx))
                    else:
                        input_views.append(native.observable_view(node_idx))

                idx = native.add_py_operator(
                    input_indices,
                    list(obs.shape),
                    dtype_str,
                    operator,
                    tuple(input_views),
                    operator.init_state(),
                )

            obs_to_idx[id(obs)] = idx

        # -- 3. Materialize requested nodes ------------------------------------
        for obs_id in self._materializations:
            if obs_id in obs_to_idx:
                native.materialize(obs_to_idx[obs_id])

        # -- 4. Build the Python source driver ---------------------------------
        driver_error: list[BaseException | None] = [None]

        async def _drive_one(
            source: Source[Any, Any],
            obs: Observable[Any, Any],
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
                    val = np.ascontiguousarray(np.asarray(raw_val, dtype=obs.dtype))
                    ts_ns = int(ts.view("int64"))
                    hist_sender.send(ts_ns, val)
                hist_sender.close()
                async for raw_val in live_iter:
                    val = np.ascontiguousarray(np.asarray(raw_val, dtype=obs.dtype))
                    live_sender.send(val)
                live_sender.close()
            except Exception:
                hist_sender.close()
                live_sender.close()
                raise

        async def _drive_all() -> None:
            tasks = [asyncio.create_task(_drive_one(src, obs, hs, ls)) for src, obs, hs, ls in source_infos]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for _, _, hs, ls in source_infos:
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

        # -- 5. Run: background thread (asyncio) + main thread (tokio) ---------
        if source_infos:
            native.run_with_driver(driver)
        else:
            native.run()

        # -- 6. Copy results back to Python series -----------------------------
        for obs_id, series in self._materializations.items():
            rust_idx = obs_to_idx[obs_id]
            n = native.series_len(rust_idx)
            if n > 0:
                ts_i64 = np.asarray(native.series_timestamps(rust_idx))
                ts_arr = ts_i64.view("datetime64[ns]")
                vals_raw = np.asarray(native.series_values(rust_idx))
                stride = math.prod(series.shape) if series.shape else 1
                if series.shape:
                    vals = vals_raw.reshape(n, *series.shape).astype(series.dtype)
                else:
                    vals = vals_raw.astype(series.dtype)
                for i in range(n):
                    series.append_unchecked(ts_arr[i], vals[i])

        # -- 7. Re-raise driver error ------------------------------------------
        if driver_error[0] is not None:
            raise driver_error[0]


def _coerce_timestamp(value: np.datetime64) -> np.datetime64:
    """Coerce a timestamp-like value to `datetime64[ns]`."""
    try:
        timestamp = np.datetime64(value)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Could not parse timestamp value {value!r}.") from exc
    return timestamp.astype("datetime64[ns]")
