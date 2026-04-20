"""General-purpose utilities shared across the package.

Python counterpart to the Rust [`tradingflow::utils`] crate module.
Currently hosts [`Schema`][tradingflow.Schema] only.

## Submodules

* [`schema`][tradingflow.utils.schema] — [`Schema`][tradingflow.Schema]
  bidirectional name ↔ position mapping.
"""

from .schema import Schema

__all__ = ["Schema"]
