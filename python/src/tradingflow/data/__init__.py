"""Primitive data types and Python-side bridge wrappers.

Python counterpart to the Rust [`tradingflow::data`] crate module.
The Rust side exposes `Array`, `Series`, `Instant`, `Duration`, the
`InputTypes` machinery, and the `Scalar` enum; on the Python side these
either appear as lightweight markers (for generic-type parameters) or
as numpy-based helpers at the FFI boundary.

## Submodules

* [`types`][tradingflow.data.types] — type markers (`Array`, `Series`,
  `Unit`), the [`NodeKind`][tradingflow.NodeKind] enum, and the
  [`Handle`][tradingflow.Handle] class used to refer to graph nodes.
* [`views`][tradingflow.data.views] — [`ArrayView`][tradingflow.ArrayView]
  and [`SeriesView`][tradingflow.SeriesView], Python wrappers over
  Rust-owned array / series buffers.
* [`time`][tradingflow.data.time] — TAI ↔ UTC conversion helpers
  ([`utc_to_tai`][tradingflow.data.time.utc_to_tai],
  [`tai_to_utc`][tradingflow.data.time.tai_to_utc]) and the
  [`coerce_timestamp`][tradingflow.data.time.coerce_timestamp] FFI
  helper.
* [`numpy`][tradingflow.data.numpy] — small numpy helpers used at the
  FFI boundary
  ([`ensure_contiguous`][tradingflow.data.numpy.ensure_contiguous]).
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
from .views import ArrayView, SeriesView
from .time import coerce_timestamp, tai_to_utc, utc_to_tai
from .numpy import ensure_contiguous

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
