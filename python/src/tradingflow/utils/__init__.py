"""General-purpose utilities shared across the package.

Python counterpart of the Rust [`tradingflow::utils`] crate module.
Currently hosts a single helper,
[`Schema`][tradingflow.utils.schema.Schema] — a bidirectional
name ↔ position mapping for labeling the axes of an array node.
"""

from .schema import Schema

__all__ = ["Schema"]
