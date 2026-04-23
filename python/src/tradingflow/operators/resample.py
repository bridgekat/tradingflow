"""Resample operator — re-emit a data input on every tick of a clock input."""

from __future__ import annotations

from .. import Handle, NativeOperator, NodeKind


class Resample(NativeOperator):
    """Re-emit `data`'s latest value on every tick of `clock`.

    The clock input's payload is never read — only its per-cycle produced
    bit is consulted — so the clock and data can be **any pair of node
    types**: `Unit`, `Array<T>`, or `Series<T>`, with independent dtypes.
    The Rust side monomorphises all nine `(data_kind × clock_kind)`
    combinations, so no kind- or dtype-matching is required at the Python
    level.

    The output matches the data input: `Resample(clock, data).kind =
    data.kind` and (for Array/Series) dtype + shape are inherited from
    `data`.

    Typical use is to align two records that would otherwise accumulate
    at heterogeneous cadences.  For example, when features depend on
    irregular financial-report updates and the training target depends
    only on trading-day prices, wrap the feature array in
    `Resample(trading_day_signal, features)` before recording it, so
    the recorded feature series and recorded target series advance
    lock-step on trading days.

    Parameters
    ----------
    clock
        Clock handle.  Only its produced bit is read — any node kind
        and dtype is accepted.
    data
        Data handle.  The output re-emits its latest value whenever
        `clock` produces; its kind, dtype, and shape are inherited.
    """

    def __init__(self, clock: Handle, data: Handle) -> None:
        # Output dtype matches data's (None for Unit, propagated through
        # NativeOperator).  Clock dtype feeds the second-level dispatch
        # inside the Rust `resample` arm and is irrelevant for Unit clocks.
        out_dtype = data.dtype
        clock_dtype = clock.dtype.name if clock.dtype is not None else None
        super().__init__(
            native_id="resample",
            inputs=(clock, data),
            kind=data.kind,
            dtype=out_dtype,
            shape=data.shape,
            params={
                "data_kind": data.kind.value,
                "clock_kind": clock.kind.value,
                "clock_dtype": clock_dtype,
            },
        )
