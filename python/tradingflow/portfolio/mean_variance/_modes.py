"""Shared Markowitz optimization mode enum (cvxpy-free)."""

from __future__ import annotations

from enum import IntEnum


class Mode(IntEnum):
    r"""Markowitz optimization mode.

    All modes share the long-only (optional) and budget constraints
    ``1^T x = 1``, ``x >= 0``.  The ``bound`` parameter's meaning is
    mode-dependent.

    - ``MIN_VARIANCE_GIVEN_RETURN``: minimize ``x^T Sigma x`` s.t. ``mu^T x >= bound``.
    - ``MAX_RETURN_GIVEN_STD_DEV``: maximize ``mu^T x`` s.t. ``sqrt(x^T Sigma x) <= bound``.
    - ``MIN_MEAN_VARIANCE``: maximize ``mu^T x - bound * x^T Sigma x``.
    - ``MIN_MEAN_STD_DEV``: maximize ``mu^T x - bound * sqrt(x^T Sigma x)``.
    """

    MIN_VARIANCE_GIVEN_RETURN = 1
    MAX_RETURN_GIVEN_STD_DEV = 2
    MIN_MEAN_VARIANCE = 3
    MIN_MEAN_STD_DEV = 4
