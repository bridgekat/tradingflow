"""Concrete mean-return predictors ported to the flowops host contract.

Leaves: ``sample``, ``single_feature``, ``linear_regression``, ``ridge``,
``lasso``.  Each exposes ``build(**kwargs)``.  The shared base lives in
``_base.py``.
"""
