"""EastMoney-specific source adapters.

This package groups data sources that follow EastMoney raw data conventions.

Subpackages
-----------
history
    Historical-market and historical-fundamental CSV source adapters.
"""

from . import history

__all__ = [
    "history",
]
