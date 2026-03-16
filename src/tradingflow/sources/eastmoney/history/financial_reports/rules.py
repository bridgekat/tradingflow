"""Rule profiles for normalizing messy raw financial report CSV columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import FinancialReportKind


@dataclass(slots=True, frozen=True)
class FinancialReportMappingProfile:
    """Normalization rules for mapping raw CSV columns to canonical field values.

    Applied in order by [`normalize_financial_report_rows`][tradingflow.sources.eastmoney.history.financial_reports.normalize_financial_report_rows] before the
    final projection step.

    Parameters
    ----------
    duplicate_items
        Mapping `{alias: canonical}`; when the canonical item is zero, its
        value is replaced by the alias item value.
    net_items
        Mapping `{net: (positive, negative)}`; reconciles a net amount
        against its positive and negative components.
    minus_items
        Raw items whose sign should be flipped (stored as expenses but
        reported as positive in the source).
    inclusion_items
        Mapping `{parent: (sub1, sub2, ...)}`; subtracts already-counted
        sub-items from the parent to avoid double-counting.
    positive_map
        Mapping `{raw_name: canonical_field_id}`; adds the raw value to
        the canonical vector field.
    negative_map
        Mapping `{raw_name: canonical_field_id}`; subtracts the raw value
        from the canonical vector field.
    metadata_columns
        Raw column names that carry metadata (dates, codes) and should not
        be treated as unknown numeric columns.
    fill_notice_delay
        Timedelta added to `report_date` when `notice_date` is missing.
    """

    duplicate_items: dict[str, str]
    net_items: dict[str, tuple[str, str]]
    minus_items: set[str]
    inclusion_items: dict[str, tuple[str, ...]]
    positive_map: dict[str, str]
    negative_map: dict[str, str]
    metadata_columns: set[str]
    fill_notice_delay: np.timedelta64

    def convert_items(self) -> set[str]:
        """Returns raw item names that should be interpreted as numeric inputs."""
        items: set[str] = set()
        items.update(self.duplicate_items.keys())
        items.update(self.duplicate_items.values())
        items.update(self.net_items.keys())
        for pos_name, neg_name in self.net_items.values():
            items.add(pos_name)
            items.add(neg_name)
        items.update(self.minus_items)
        items.update(self.inclusion_items.keys())
        for subitems in self.inclusion_items.values():
            items.update(subitems)
        items.update(self.positive_map.keys())
        items.update(self.negative_map.keys())
        return items


_COMMON_METADATA_COLUMNS: set[str] = {
    "NOTICE_DATE",
    "UPDATE_DATE",
    "CURRENCY",
    "SECURITY_TYPE_CODE",
    "SECURITY_CODE",
    "SECURITY_NAME_ABBR",
    "SECUCODE",
    "ORG_TYPE",
    "ORG_CODE",
    "REPORT_TYPE",
    "REPORT_DATE",
    "REPORT_DATE_NAME",
    "OPINION_TYPE",
    "OSOPINION_TYPE",
    "OSOOPINION_TYPE",
    "LISTING_STATE",
}


BALANCE_SHEET_MAPPING_PROFILE = FinancialReportMappingProfile(
    duplicate_items={
        "TRADE_FINASSET_NOTFVTPL": "TRADE_FINASSET",
        "TRADE_FINLIAB_NOTFVTPL": "TRADE_FINLIAB",
        "SHORT_FIN_PAYABLE": "SHORT_BOND_PAYABLE",
        "ADVANCE_RECE": "ADVANCE_RECEIVABLES",
    },
    net_items={
        "NET_PENDMORTGAGE_ASSET": ("PEND_MORTGAGE_ASSET", "MORTGAGE_ASSET_IMPAIRMENT"),
    },
    minus_items={
        "MORTGAGE_ASSET_IMPAIRMENT",
        "TREASURY_SHARES",
        "UNCONFIRM_INVEST_LOSS",
    },
    inclusion_items={
        "FVTPL_FINASSET": ("TRADE_FINASSET", "APPOINT_FVTPL_FINASSET"),
        "FVTPL_FINLIAB": ("TRADE_FINLIAB", "APPOINT_FVTPL_FINLIAB"),
    },
    positive_map={
        "TOTAL_ASSETS": "balance_sheet.assets",
        "TRADE_FINASSET": "balance_sheet.assets.current.financial.fvpl.trading",
        "APPOINT_FVTPL_FINASSET": "balance_sheet.assets.current.financial.fvpl.other",
        "FVTPL_FINASSET": "balance_sheet.assets.current.financial.fvpl.other",
        "PEND_MORTGAGE_ASSET": "balance_sheet.assets.current.other",
        "MORTGAGE_ASSET_IMPAIRMENT": "balance_sheet.assets.current.other",
    },
    negative_map={
        "TOTAL_LIABILITIES": "balance_sheet.liab",
        "TOTAL_EQUITY": "balance_sheet.equity",
        "TRADE_FINLIAB": "balance_sheet.liab.current.financial.fvpl.trading",
        "APPOINT_FVTPL_FINLIAB": "balance_sheet.liab.current.financial.fvpl.other",
        "FVTPL_FINLIAB": "balance_sheet.liab.current.financial.fvpl.other",
    },
    metadata_columns=_COMMON_METADATA_COLUMNS,
    fill_notice_delay=np.timedelta64(45, "D"),
)


INCOME_STATEMENT_MAPPING_PROFILE = FinancialReportMappingProfile(
    duplicate_items={
        "FAIRVALUE_CHANGE": "FAIRVALUE_CHANGE_INCOME",
    },
    net_items={
        "INTEREST_NI": ("INTEREST_INCOME", "INTEREST_EXPENSE"),
        "FEE_COMMISSION_NI": ("FEE_COMMISSION_INCOME", "FEE_COMMISSION_EXPENSE"),
    },
    minus_items={
        "AMORTIZE_COMPENSATE_EXPENSE",
        "AMORTIZE_INSURANCE_RESERVE",
        "AMORTIZE_REINSURE_EXPENSE",
        "FE_INTEREST_INCOME",
    },
    inclusion_items={
        "INVEST_INCOME": ("INVEST_JOINT_INCOME", "ACF_END_INCOME"),
        "FINANCE_EXPENSE": ("FE_INTEREST_EXPENSE", "FE_INTEREST_INCOME"),
    },
    positive_map={
        "TOTAL_OPERATE_INCOME": "income_statement.profit.operating.income",
        "OPERATE_PROFIT": "income_statement.profit.operating",
        "NONBUSINESS_INCOME": "income_statement.profit.other_income",
        "NETPROFIT": "income_statement.profit",
        "PARENT_NETPROFIT": "income_statement.parent_interests",
        "MINORITY_INTEREST": "income_statement.noncontrolling_interests",
        "NETPROFIT_OTHER": "income_statement.other",
        "INVEST_INCOME": "income_statement.profit.operating.income.investment.other",
        "FAIRVALUE_CHANGE_INCOME": "income_statement.profit.operating.income.investment.fvpl",
    },
    negative_map={
        "TOTAL_OPERATE_COST": "income_statement.profit.operating.expenses",
        "NONBUSINESS_EXPENSE": "income_statement.profit.other_expenses",
        "INCOME_TAX": "income_statement.profit.income_taxes",
    },
    metadata_columns=_COMMON_METADATA_COLUMNS,
    fill_notice_delay=np.timedelta64(45, "D"),
)


def default_mapping_profile(kind: FinancialReportKind) -> FinancialReportMappingProfile:
    """Returns the default mapping profile for *kind*."""
    if kind == "balance_sheet":
        return BALANCE_SHEET_MAPPING_PROFILE
    if kind == "income_statement":
        return INCOME_STATEMENT_MAPPING_PROFILE
    raise ValueError(f"Unsupported financial report kind: {kind!r}")


def map_with_defaults(data: dict[str, Any] | None) -> dict[str, Any]:
    """Returns a shallow dict copy from an optional mapping."""
    return dict(data or {})


__all__ = [
    "BALANCE_SHEET_MAPPING_PROFILE",
    "INCOME_STATEMENT_MAPPING_PROFILE",
    "FinancialReportMappingProfile",
    "default_mapping_profile",
]
