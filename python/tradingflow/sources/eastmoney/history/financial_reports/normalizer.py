"""Normalization pipeline for raw financial report CSV rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rules import FinancialReportMappingProfile
from .schema import FinancialReportKind, FinancialReportSchema


@dataclass(slots=True, frozen=True)
class FinancialReportRow:
    """One normalized financial report row ready for source emission."""

    report_date: np.datetime64
    notice_date: np.datetime64
    relevance_date: np.datetime64
    values: np.ndarray[tuple[int], np.dtype[np.float64]]
    error_flag: bool


@dataclass(slots=True, frozen=True)
class FinancialReportDiagnostics:
    """Normalization diagnostics collected from raw input rows."""

    unknown_columns: tuple[str, ...]
    equation_failures: tuple[np.datetime64, ...]
    dropped_rows: int
    revision_dropped_rows: int
    total_rows: int
    emitted_rows: int

    @staticmethod
    def empty() -> FinancialReportDiagnostics:
        """Returns an empty diagnostics record."""
        return FinancialReportDiagnostics(
            unknown_columns=(),
            equation_failures=(),
            dropped_rows=0,
            revision_dropped_rows=0,
            total_rows=0,
            emitted_rows=0,
        )


@dataclass(slots=True)
class _ParsedRawRow:
    report_date: np.datetime64
    notice_date: np.datetime64
    relevance_date: np.datetime64
    raw_values: dict[str, float]


def normalize_financial_report_rows(
    rows: list[dict[str, str]],
    *,
    kind: FinancialReportKind,
    schema: FinancialReportSchema,
    mapping_profile: FinancialReportMappingProfile,
    symbol: str | None = None,
    strict_unknown_columns: bool = False,
    strict_equation_check: bool = False,
) -> tuple[list[FinancialReportRow], FinancialReportDiagnostics]:
    """Normalizes raw CSV rows into canonical report rows."""
    _validate_schema_targets(schema, mapping_profile)

    convert_items = mapping_profile.convert_items()
    unknown_columns: set[str] = set()
    parsed_rows: list[_ParsedRawRow] = []
    dropped_rows = 0

    for row in rows:
        report_date = _parse_datetime64_ns(row.get("REPORT_DATE"))
        if report_date is None:
            dropped_rows += 1
            continue

        currency = (row.get("CURRENCY") or "").strip()
        if currency and currency != "CNY":
            dropped_rows += 1
            continue

        update_date = _parse_datetime64_ns(row.get("UPDATE_DATE"))
        notice_date = update_date or _parse_datetime64_ns(row.get("NOTICE_DATE"))
        if notice_date is None:
            notice_date = report_date + mapping_profile.fill_notice_delay

        relevance_date = report_date if report_date >= notice_date else notice_date

        raw_values = {item: _parse_number(row.get(item)) for item in convert_items}
        parsed_rows.append(
            _ParsedRawRow(
                report_date=report_date,
                notice_date=notice_date,
                relevance_date=relevance_date,
                raw_values=raw_values,
            )
        )

        for column_name in row:
            if column_name.endswith("_YOY") or column_name.endswith("_BALANCE"):
                continue
            if column_name in convert_items or column_name in mapping_profile.metadata_columns:
                continue
            unknown_columns.add(column_name)

    if unknown_columns and strict_unknown_columns:
        symbol_suffix = f" for symbol {symbol}" if symbol else ""
        raise ValueError(
            f"Found unmapped columns{symbol_suffix}: {sorted(unknown_columns)}"
        )

    revision_dropped_rows = 0
    by_report_date: dict[np.datetime64, _ParsedRawRow] = {}
    for row in parsed_rows:
        existing = by_report_date.get(row.report_date)
        if existing is None:
            by_report_date[row.report_date] = row
            continue
        if row.notice_date > existing.notice_date:
            by_report_date[row.report_date] = row
            revision_dropped_rows += 1
        else:
            revision_dropped_rows += 1

    collapsed = sorted(
        by_report_date.values(),
        key=lambda row: (row.relevance_date, row.notice_date, row.report_date),
    )

    unique_relevance_rows: list[_ParsedRawRow] = []
    for row in collapsed:
        if unique_relevance_rows and row.relevance_date == unique_relevance_rows[-1].relevance_date:
            unique_relevance_rows[-1] = row
            revision_dropped_rows += 1
        else:
            unique_relevance_rows.append(row)

    normalized_rows: list[FinancialReportRow] = []
    equation_failures: list[np.datetime64] = []

    for parsed_row in unique_relevance_rows:
        vector = _project_values(
            raw_values=parsed_row.raw_values,
            schema=schema,
            mapping_profile=mapping_profile,
        )
        error_flag = _check_equation(kind=kind, schema=schema, vector=vector)
        if error_flag:
            equation_failures.append(parsed_row.relevance_date)
            if strict_equation_check:
                symbol_suffix = f" for symbol {symbol}" if symbol else ""
                raise ValueError(
                    f"Equation check failed at {parsed_row.relevance_date!r}{symbol_suffix}."
                )

        vector[vector == 0.0] = np.nan
        normalized_rows.append(
            FinancialReportRow(
                report_date=parsed_row.report_date,
                notice_date=parsed_row.notice_date,
                relevance_date=parsed_row.relevance_date,
                values=vector,
                error_flag=error_flag,
            )
        )

    diagnostics = FinancialReportDiagnostics(
        unknown_columns=tuple(sorted(unknown_columns)),
        equation_failures=tuple(equation_failures),
        dropped_rows=dropped_rows,
        revision_dropped_rows=revision_dropped_rows,
        total_rows=len(rows),
        emitted_rows=len(normalized_rows),
    )
    return normalized_rows, diagnostics


def _project_values(
    *,
    raw_values: dict[str, float],
    schema: FinancialReportSchema,
    mapping_profile: FinancialReportMappingProfile,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    raw = dict(raw_values)

    for alias_name, canonical_name in mapping_profile.duplicate_items.items():
        if raw.get(canonical_name, 0.0) == 0.0:
            raw[canonical_name] = raw.get(canonical_name, 0.0) + raw.get(alias_name, 0.0)

    for net_name, (positive_name, negative_name) in mapping_profile.net_items.items():
        net_adjust = raw.get(net_name, 0.0) - (raw.get(positive_name, 0.0) - raw.get(negative_name, 0.0))
        if net_adjust >= 0.0:
            raw[positive_name] = raw.get(positive_name, 0.0) + net_adjust
        else:
            raw[negative_name] = raw.get(negative_name, 0.0) - net_adjust

    for minus_name in mapping_profile.minus_items:
        raw[minus_name] = raw.get(minus_name, 0.0) * -1.0

    for parent_name, subitems in mapping_profile.inclusion_items.items():
        parent_value = raw.get(parent_name, 0.0)
        if parent_value != 0.0:
            raw[parent_name] = parent_value - sum(raw.get(subitem, 0.0) for subitem in subitems)

    vector = np.zeros(len(schema.field_ids), dtype=np.float64)
    for raw_name, field_id in mapping_profile.positive_map.items():
        vector[schema.field_index[field_id]] += raw.get(raw_name, 0.0)
    for raw_name, field_id in mapping_profile.negative_map.items():
        vector[schema.field_index[field_id]] -= raw.get(raw_name, 0.0)
    return vector


def _check_equation(
    *,
    kind: FinancialReportKind,
    schema: FinancialReportSchema,
    vector: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> bool:
    if kind == "balance_sheet":
        assets = _field_or_zero(schema, vector, "balance_sheet.assets")
        liab = _field_or_zero(schema, vector, "balance_sheet.liab")
        equity = _field_or_zero(schema, vector, "balance_sheet.equity")
        residual = assets + liab + equity
        _set_if_present(schema, vector, "balance_sheet.residual", residual)
        return _is_significant_residual(residual, assets)

    if kind == "income_statement":
        operating = _field_or_zero(schema, vector, "income_statement.profit.operating")
        operating_income = _field_or_zero(schema, vector, "income_statement.profit.operating.income")
        operating_expenses = _field_or_zero(schema, vector, "income_statement.profit.operating.expenses")
        operating_other = _field_or_zero(schema, vector, "income_statement.profit.operating.other")
        operating_residual = operating - (operating_income + operating_expenses + operating_other)
        _set_if_present(schema, vector, "income_statement.profit.operating.residual", operating_residual)

        profit = _field_or_zero(schema, vector, "income_statement.profit")
        other_income = _field_or_zero(schema, vector, "income_statement.profit.other_income")
        other_expenses = _field_or_zero(schema, vector, "income_statement.profit.other_expenses")
        income_taxes = _field_or_zero(schema, vector, "income_statement.profit.income_taxes")
        profit_residual = profit - (operating + other_income + other_expenses + income_taxes)
        _set_if_present(schema, vector, "income_statement.profit.residual", profit_residual)

        parent = _field_or_zero(schema, vector, "income_statement.parent_interests")
        minority = _field_or_zero(schema, vector, "income_statement.noncontrolling_interests")
        other = _field_or_zero(schema, vector, "income_statement.other")
        statement_residual = profit - (parent + minority + other)
        _set_if_present(schema, vector, "income_statement.residual", statement_residual)

        return any(
            (
                _is_significant_residual(statement_residual, profit),
                _is_significant_residual(profit_residual, profit),
                _is_significant_residual(operating_residual, operating),
            )
        )

    raise ValueError(f"Unsupported financial report kind: {kind!r}")


def _field_or_zero(
    schema: FinancialReportSchema,
    vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    field_id: str,
) -> float:
    index = schema.field_index.get(field_id)
    if index is None:
        return 0.0
    return float(vector[index])


def _set_if_present(
    schema: FinancialReportSchema,
    vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    field_id: str,
    value: float,
) -> None:
    index = schema.field_index.get(field_id)
    if index is not None:
        vector[index] = value


def _is_significant_residual(residual: float, base: float) -> bool:
    if abs(residual) < 0.01:
        return False
    if base == 0.0:
        return True
    return abs(residual / base) >= 0.01


def _parse_datetime64_ns(value: str | None) -> np.datetime64 | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return np.datetime64(text).astype("datetime64[ns]")
    except Exception:
        return None


def _parse_number(value: str | None) -> float:
    if value is None:
        return 0.0
    text = value.strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except Exception:
        return 0.0


def _validate_schema_targets(
    schema: FinancialReportSchema,
    mapping_profile: FinancialReportMappingProfile,
) -> None:
    missing_targets = {
        *mapping_profile.positive_map.values(),
        *mapping_profile.negative_map.values(),
    } - set(schema.field_ids)
    if missing_targets:
        raise ValueError(
            "Mapping profile references unknown canonical fields: "
            f"{sorted(missing_targets)}"
        )


__all__ = [
    "FinancialReportDiagnostics",
    "FinancialReportRow",
    "normalize_financial_report_rows",
]
