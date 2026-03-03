"""Time series operators.

This package contains all concrete operator implementations, organised
into sub-packages by category:

* **ops** (this entry point) – :class:`Apply` (generic n-ary function
  application) and element-wise arithmetic factories (:func:`add`,
  :func:`subtract`, :func:`multiply`, :func:`divide`, :func:`negate`,
  :func:`multiple`).
* **ops.filters** – Rolling statistical filters and technical indicators.
* **ops.metrics** – Performance metrics (average return, Sharpe ratio).
* **ops.portfolios** – Portfolio construction methods.
* **ops.predictors** – Prediction models with rolling retraining.
* **ops.signals** – Trading signal generators (placeholder).
* **ops.simulators** – Trading simulators.
"""

from .apply import Apply, add, subtract, multiply, divide, multiple, negate


__all__ = [
    "Apply",
    "add",
    "subtract",
    "multiply",
    "divide",
    "multiple",
    "negate",
]
