"""Primitive data types and Python-side bridge wrappers.

This module is the Python counterpart of the Rust
[`tradingflow::data`] crate module.  The Rust side owns the actual
storage of every node's value (`Array<T>`, `Series<T>`, `Instant`,
`Duration`, plus the `InputTypes` and `Scalar` machinery).  Across the
FFI boundary those values appear here as:

- **Type markers** — zero-sized classes that encode a node's value
  kind in the type system (e.g. `Handle[Array[np.float64]]`), used
  purely for static typing.
- **Views** — thin Python wrappers around Rust-owned buffers that let
  Python operators read and write node values without copying.
- **Time helpers** — small utilities for converting between TAI (the
  internal wire format) and UTC at program boundaries.

Most users never import from this module directly — the commonly used
names ([`Handle`][tradingflow.data.types.Handle], [`Array`][tradingflow.data.types.Array],
[`Series`][tradingflow.data.types.Series], [`ArrayView`][tradingflow.data.views.ArrayView],
[`SeriesView`][tradingflow.data.views.SeriesView], [`NodeKind`][tradingflow.data.types.NodeKind],
[`Unit`][tradingflow.data.types.Unit]) are re-exported at the package root.

## Sub-modules

- [`types`][tradingflow.data.types] — type markers and
  [`Handle`][tradingflow.data.types.Handle], the typed reference returned by
  `Scenario.add_source` / `add_operator`.
- [`views`][tradingflow.data.views] —
  [`ArrayView`][tradingflow.data.views.ArrayView] and
  [`SeriesView`][tradingflow.data.views.SeriesView], plus the
  [`ensure_contiguous`][tradingflow.data.views.ensure_contiguous]
  helper that normalizes NumPy inputs at the FFI edge.
- [`time`][tradingflow.data.time] — TAI ↔ UTC conversion helpers
  ([`utc_to_tai`][tradingflow.data.time.utc_to_tai],
  [`tai_to_utc`][tradingflow.data.time.tai_to_utc]) and the
  [`coerce_timestamp`][tradingflow.data.time.coerce_timestamp] FFI
  helper.  See the timestamp section in
  [`tradingflow`][tradingflow] for the reasoning behind the TAI choice.
"""

from .types import (
    Array,
    Handle,
    NodeKind,
    Series,
    Unit,
    _to_native_node_kind,
    node_type_to_name,
)
from .views import ArrayView, SeriesView, ensure_contiguous
from .time import coerce_timestamp, tai_to_utc, utc_to_tai

__all__ = [
    "Array",
    "ArrayView",
    "Handle",
    "NodeKind",
    "Series",
    "SeriesView",
    "Unit",
    "coerce_timestamp",
    "ensure_contiguous",
    "node_type_to_name",
    "tai_to_utc",
    "utc_to_tai",
]
