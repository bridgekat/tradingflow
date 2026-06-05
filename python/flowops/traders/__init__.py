"""Trading simulation operators (ported to the flowops host contract).

Each leaf module exposes ``build(**kwargs) -> op`` implementing the host
contract.  Shared base state / compute logic lives in ``_base.py``.
"""
