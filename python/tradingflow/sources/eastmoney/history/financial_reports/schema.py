"""Canonical schema definitions for EastMoney financial report source vectors.

This module defines stable canonical field orders for normalized financial
report vectors and provides schema lookup helpers. The field order is part of
the public contract for vector-valued source series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


type FinancialReportKind = Literal["balance_sheet", "income_statement"]


@dataclass(slots=True, frozen=True)
class FinancialReportSchema:
    """Canonical schema for one financial report kind.

    Parameters
    ----------
    field_ids
        Canonical field identifiers in stable vector order.
    field_index
        Mapping from canonical field identifier to vector index.
    """

    field_ids: tuple[str, ...]
    field_index: dict[str, int]

    @classmethod
    def from_field_ids(cls, field_ids: tuple[str, ...]) -> FinancialReportSchema:
        """Builds a schema from ordered field identifiers."""
        if not field_ids:
            raise ValueError("field_ids must not be empty.")
        if len(set(field_ids)) != len(field_ids):
            raise ValueError("field_ids must be unique.")
        return cls(field_ids=field_ids, field_index={field_id: i for i, field_id in enumerate(field_ids)})


BALANCE_SHEET_SCHEMA = FinancialReportSchema.from_field_ids(
    (
        "balance_sheet.assets",
        "balance_sheet.liab",
        "balance_sheet.equity",
        "balance_sheet.residual",
        "balance_sheet.assets.current.financial.fvpl.trading",
        "balance_sheet.assets.current.financial.fvpl.other",
        "balance_sheet.assets.current.other",
        "balance_sheet.liab.current.financial.fvpl.trading",
        "balance_sheet.liab.current.financial.fvpl.other",
    )
)

INCOME_STATEMENT_SCHEMA = FinancialReportSchema.from_field_ids(
    (
        "income_statement.profit",
        "income_statement.profit.residual",
        "income_statement.profit.operating",
        "income_statement.profit.operating.residual",
        "income_statement.profit.operating.income",
        "income_statement.profit.operating.expenses",
        "income_statement.profit.operating.other",
        "income_statement.profit.other_income",
        "income_statement.profit.other_expenses",
        "income_statement.profit.income_taxes",
        "income_statement.parent_interests",
        "income_statement.noncontrolling_interests",
        "income_statement.other",
        "income_statement.residual",
        "income_statement.profit.operating.income.investment.other",
        "income_statement.profit.operating.income.investment.fvpl",
    )
)


def default_schema(kind: FinancialReportKind) -> FinancialReportSchema:
    """Returns the default canonical schema for *kind*."""
    if kind == "balance_sheet":
        return BALANCE_SHEET_SCHEMA
    if kind == "income_statement":
        return INCOME_STATEMENT_SCHEMA
    raise ValueError(f"Unsupported financial report kind: {kind!r}")


__all__ = [
    "BALANCE_SHEET_SCHEMA",
    "INCOME_STATEMENT_SCHEMA",
    "FinancialReportKind",
    "FinancialReportSchema",
    "default_schema",
]
