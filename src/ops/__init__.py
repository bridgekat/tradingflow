"""Time series operators.

This package contains all concrete operator implementations, organized into
sub-packages by category:

* **ops** (this entry point): :class:`Apply`, arithmetic factories
  (:func:`add`, :func:`subtract`, :func:`multiply`, :func:`divide`,
  :func:`negate`), and :class:`Select`.
* **ops.filters**: Rolling statistical filters and technical indicators.
* **ops.metrics**: Performance metrics (average return, Sharpe ratio).
* **ops.portfolios**: Portfolio construction methods.
* **ops.predictors**: Prediction models with rolling retraining.
* **ops.signals**: Trading signal generators (placeholder).
* **ops.simulators**: Trading simulators.
"""

from .apply import Apply, add, divide, multiply, negate, subtract
from .select import Select, select

__all__ = [
    "Apply",
    "Select",
    "add",
    "divide",
    "multiply",
    "negate",
    "select",
    "subtract",
]
