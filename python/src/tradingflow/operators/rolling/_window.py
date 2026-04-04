"""Internal utilities for rolling operators."""

from __future__ import annotations

import numpy as np


def window_params(window: int | np.timedelta64) -> dict:
    """Convert a window argument to native params dict."""
    if isinstance(window, np.timedelta64):
        window_ns = int(np.timedelta64(window, "ns").astype(np.int64))
        return {"window_ns": window_ns}
    return {"window": int(window)}
