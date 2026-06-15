"""Python operators for the flowgraph engine (loaded by the Rust `pyhost`).

Each operator module defines `build(**kwargs) -> op`, where `op` implements the
host contract:

    init(self, inputs, timestamp) -> state
    @staticmethod
    def compute(state, inputs, output, timestamp, produced) -> bool

`inputs` is a tuple of views (array views with `.value()`/`.to_numpy()`, series
views with `.values(start, end)`/`.last()`/`.at(i)`/`len()`/`[-1]`), `output` is
a writable array view (`.write(ndarray)`), `timestamp` is TAI nanoseconds, and
`produced` is a `tuple[bool, ...]` parallel to `inputs`. `np`/`numpy` are
available; `scipy` may be imported. These mirror the legacy
`tradingflow.operator.Operator` contract so operators port nearly verbatim.
"""
