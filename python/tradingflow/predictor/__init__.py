"""Prediction models, as Python operators hosted by the Rust engine.

Each module defines a operator whose contract mirrors the Rust `Operator`
trait, and binds it to `__op__` (or defines `build(**kwargs)` when the
operator takes construction parameters):

    init(self, inputs) -> state
    reset(inputs, state) -> outputs
    compute(inputs, state, instant) -> outputs

See `tradingflow.metric` for the host contract in full.

# Recording history

Predictors fit on a window of past cross-sections, which under the arrays-only
boundary they must record themselves: every leaf arrives as the *latest*
cross-section, never as a series. A predictor therefore appends each sampled
pair to a bounded deque of its own rather than reading back through an upstream
`Record`.

`max_periods` sizes that window: the deque holds `max_periods + target_offset`
cross-sections, enough to form `max_periods` pairs once the forward offset is
accounted for. `max_periods=None` keeps every pair, and the memory then grows
without bound; prefer a window unless the run is short.

What actually costs memory is the *feature* panel, which is `F` times wider
than the target. So retaining it is opt-in per model, via `retain_features`:

* The windowed regressions (`linear_regression`, `ridge`, `lasso`) fit on a
  panel and retain it.
* The covariance predictors and `sample` fit from the target alone and retain
  no features at all — and so do not withhold a stock whose features are
  unusable, having never looked at them.
* The incremental regressions hold only the `target_offset + 1` cross-sections
  needed to form the next pair, whatever the window, because each pair is
  folded into a sufficient-statistic pool as soon as its forward target lands.
  A rolling window there retains each period's moment *contribution* for
  down-dating, which is `O(F**2)` rather than `O(N F)`.
"""
