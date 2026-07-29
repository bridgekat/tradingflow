"""Python operators for the TradingFlow engine (loaded by the Rust `pyhost`).

Each operator module defines `build(**kwargs) -> op`, where `op` implements the
host contract (the argument order mirrors the Rust `Segment` trait):

    init(self, inputs) -> state
    @staticmethod
    def compute(inputs, state, timestamp) -> ndarray | None

`inputs` is a single tuple with one entry per wired leaf, in wiring order:
array views (`.value()`/`.to_numpy()`), series views
(`.values(start, end)`/`.last()`/`.at(i)`/`len()`/`[-1]`), and signal views for
signal leaves of any rank (a rank-0 pulse answers `if signal:` directly; a
signal array reads as a NumPy bool mask via `.value()`). `timestamp` is TAI
nanoseconds.

`compute` **returns** this generation's output batch, like `Segment::compute`
does: an array pulses the output signal, `None` is quiescence (the values wire
retains its last batch). The return is normalized with
`numpy.ascontiguousarray`, so a list or a plain float is accepted, and it must
match the output's element count. `np`/`numpy` are available; `scipy` may be
imported.
"""
