"""Concrete covariance predictors ported to the pyhost contract.

Leaves: ``sample``, ``single_index``, ``shrinkage``, ``rmt`` (RMT0/RMTM),
``hierarchical`` (UPGMA/WPGMA/Hausdorff).  Each exposes ``build(**kwargs)``.
The shared base lives in ``_base.py`` and shared estimators in ``_common.py``.
"""
