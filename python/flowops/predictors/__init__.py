"""Cross-sectional mean and covariance predictors, ported to the flowops host.

Each concrete leaf module (`mean/*.py`, `variance/*.py`) exposes
``build(**kwargs)`` returning an operator implementing the host contract
(``init`` / ``compute``).  The shared abstract bases live in
``predictors/mean/_base.py`` and ``predictors/variance/_base.py``.
"""
