"""Portfolio construction, as Python operators hosted by the Rust engine.

Each module defines a operator whose contract mirrors the Rust `Operator`
trait, and binds it to `__op__` (or defines `build(**kwargs)` when the
operator takes construction parameters):

    init(self, inputs) -> state
    reset(inputs, state) -> outputs
    compute(inputs, state, instant) -> outputs

See `tradingflow.metric` for the host contract in full.

# Log versus linear returns

Predictors upstream work in log returns, because those are what aggregate
additively over time. Optimizers here work in linear returns, because those
are what aggregate additively across a portfolio — a weighted sum of log
returns is not the log return of the weighted sum. The `logarithmic` flag
puts the lognormal moment map between them, so a portfolio is optimizing the
quantity it actually reports.

# The active subset

Only stocks in the universe with finite moments are optimized over. A solver
sees the masked-down problem, and the weights it returns are scattered back;
everything else is zero. Sizing solvers to `max_universe_size` rather than the
full cross-section is what keeps the warm start affordable.
"""
