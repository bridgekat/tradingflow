"""Time series operators.

This package contains all concrete operator implementations, organized into
sub-packages by category:

* **operators** (this entry point): :class:`Apply`, arithmetic factories
  (:func:`add`, :func:`subtract`, :func:`multiply`, :func:`divide`,
  :func:`negate`), :class:`Concat`, :class:`Filter`, :class:`Stack`,
  :class:`Where`, and :class:`Select`.
* **operators.filters**: Rolling statistical filters and technical indicators.
* **operators.metrics**: Performance metrics (average return, Sharpe ratio).
* **operators.portfolios**: Portfolio construction methods.
* **operators.predictors**: Prediction models with rolling retraining.
* **operators.signals**: Trading signal generators (placeholder).
* **operators.simulators**: Trading simulators.

:class:`Filter` drops entire elements when a scalar predicate returns
``False``.  :class:`Where` replaces individual array elements with a fill
value (element-wise conditional, analogous to ``np.where``).
"""

from .apply import Apply, add, divide, map, multiply, negate, subtract
from .concat import Concat
from .filter import Filter
from .stack import Stack
from .select import Select, select
from .where import Where

__all__ = [
    "Apply",
    "Concat",
    "Filter",
    "Select",
    "Stack",
    "Where",
    "add",
    "divide",
    "map",
    "multiply",
    "negate",
    "select",
    "subtract",
]
