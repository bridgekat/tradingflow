"""Tests for financial report normalization and source integration."""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import numpy as np
import pytest

from src import (
    ArrayBundleSource,
    Scenario,
)
from src.ops import add, select
from src.sources.eastmoney.history import (
    BALANCE_SHEET_MAPPING_PROFILE,
    BALANCE_SHEET_SCHEMA,
    INCOME_STATEMENT_MAPPING_PROFILE,
    INCOME_STATEMENT_SCHEMA,
    FinancialReportCSVSource,
    normalize_financial_report_rows,
)


def ts_iso(text: str) -> np.datetime64:
    """Helper to create datetime64[ns] from ISO-like text."""
    return np.datetime64(text).astype("datetime64[ns]")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestSchema:
    def test_default_schema_field_order_and_index(self) -> None:
        for schema in (BALANCE_SHEET_SCHEMA, INCOME_STATEMENT_SCHEMA):
            assert len(schema.field_ids) == len(schema.field_index)
            assert len(set(schema.field_ids)) == len(schema.field_ids)
            for i, field_id in enumerate(schema.field_ids):
                assert schema.field_index[field_id] == i


class TestNormalizer:
    def test_balance_rules_alias_net_inclusion(self) -> None:
        rows = [
            {
                "SECUCODE": "000001.SZ",
                "REPORT_DATE": "2021-12-31 00:00:00",
                "NOTICE_DATE": "2022-03-20 00:00:00",
                "UPDATE_DATE": "",
                "CURRENCY": "CNY",
                "REPORT_TYPE": "年报",
                "TOTAL_ASSETS": "100",
                "TOTAL_LIABILITIES": "60",
                "TOTAL_EQUITY": "40",
                "TRADE_FINASSET": "0",
                "TRADE_FINASSET_NOTFVTPL": "8",
                "APPOINT_FVTPL_FINASSET": "1",
                "FVTPL_FINASSET": "20",
                "PEND_MORTGAGE_ASSET": "10",
                "MORTGAGE_ASSET_IMPAIRMENT": "2",
                "NET_PENDMORTGAGE_ASSET": "3",
                "TRADE_FINLIAB": "4",
                "TRADE_FINLIAB_NOTFVTPL": "2",
                "APPOINT_FVTPL_FINLIAB": "1",
                "FVTPL_FINLIAB": "10",
                "MYSTERY_BALANCE_FIELD": "1",
            }
        ]

        normalized_rows, diagnostics = normalize_financial_report_rows(
            rows,
            kind="balance_sheet",
            schema=BALANCE_SHEET_SCHEMA,
            mapping_profile=BALANCE_SHEET_MAPPING_PROFILE,
        )

        assert len(normalized_rows) == 1
        assert diagnostics.unknown_columns == ("MYSTERY_BALANCE_FIELD",)

        values = normalized_rows[0].values
        i = BALANCE_SHEET_SCHEMA.field_index
        assert float(values[i["balance_sheet.assets.current.financial.fvpl.trading"]]) == pytest.approx(8.0)
        assert float(values[i["balance_sheet.assets.current.financial.fvpl.other"]]) == pytest.approx(12.0)
        assert float(values[i["balance_sheet.assets.current.other"]]) == pytest.approx(3.0)
        assert normalized_rows[0].error_flag is False

    def test_unknown_column_can_be_strict(self) -> None:
        rows = [
            {
                "SECUCODE": "000001.SZ",
                "REPORT_DATE": "2021-12-31 00:00:00",
                "NOTICE_DATE": "2022-03-20 00:00:00",
                "UPDATE_DATE": "",
                "CURRENCY": "CNY",
                "REPORT_TYPE": "年报",
                "TOTAL_ASSETS": "100",
                "TOTAL_LIABILITIES": "60",
                "TOTAL_EQUITY": "40",
                "UNMAPPED_FIELD": "1",
            }
        ]
        with pytest.raises(ValueError, match="unmapped columns"):
            normalize_financial_report_rows(
                rows,
                kind="balance_sheet",
                schema=BALANCE_SHEET_SCHEMA,
                mapping_profile=BALANCE_SHEET_MAPPING_PROFILE,
                strict_unknown_columns=True,
            )

    def test_revision_collapse_uses_latest_notice(self) -> None:
        rows = [
            {
                "SECUCODE": "000001.SZ",
                "REPORT_DATE": "2021-12-31 00:00:00",
                "NOTICE_DATE": "2022-03-10 00:00:00",
                "UPDATE_DATE": "",
                "CURRENCY": "CNY",
                "REPORT_TYPE": "年报",
                "TOTAL_ASSETS": "100",
                "TOTAL_LIABILITIES": "60",
                "TOTAL_EQUITY": "40",
            },
            {
                "SECUCODE": "000001.SZ",
                "REPORT_DATE": "2021-12-31 00:00:00",
                "NOTICE_DATE": "2022-03-20 00:00:00",
                "UPDATE_DATE": "",
                "CURRENCY": "CNY",
                "REPORT_TYPE": "年报",
                "TOTAL_ASSETS": "110",
                "TOTAL_LIABILITIES": "70",
                "TOTAL_EQUITY": "40",
            },
        ]
        normalized_rows, diagnostics = normalize_financial_report_rows(
            rows,
            kind="balance_sheet",
            schema=BALANCE_SHEET_SCHEMA,
            mapping_profile=BALANCE_SHEET_MAPPING_PROFILE,
        )

        assert len(normalized_rows) == 1
        assert diagnostics.revision_dropped_rows == 1
        assert normalized_rows[0].relevance_date == ts_iso("2022-03-20 00:00:00")
        assets_index = BALANCE_SHEET_SCHEMA.field_index["balance_sheet.assets"]
        assert float(normalized_rows[0].values[assets_index]) == pytest.approx(110.0)

    def test_income_equation_failures_can_be_strict(self) -> None:
        rows = [
            {
                "SECUCODE": "000001.SZ",
                "REPORT_DATE": "2021-12-31 00:00:00",
                "NOTICE_DATE": "2022-03-20 00:00:00",
                "UPDATE_DATE": "",
                "CURRENCY": "CNY",
                "REPORT_TYPE": "年报",
                "TOTAL_OPERATE_INCOME": "200",
                "TOTAL_OPERATE_COST": "120",
                "OPERATE_PROFIT": "80",
                "NONBUSINESS_INCOME": "5",
                "NONBUSINESS_EXPENSE": "1",
                "INCOME_TAX": "20",
                "NETPROFIT": "50",
                "PARENT_NETPROFIT": "46",
                "MINORITY_INTEREST": "3",
                "NETPROFIT_OTHER": "1",
            }
        ]

        normalized_rows, diagnostics = normalize_financial_report_rows(
            rows,
            kind="income_statement",
            schema=INCOME_STATEMENT_SCHEMA,
            mapping_profile=INCOME_STATEMENT_MAPPING_PROFILE,
        )
        assert len(normalized_rows) == 1
        assert normalized_rows[0].error_flag is True
        assert diagnostics.equation_failures == (ts_iso("2022-03-20 00:00:00"),)

        with pytest.raises(ValueError, match="Equation check failed"):
            normalize_financial_report_rows(
                rows,
                kind="income_statement",
                schema=INCOME_STATEMENT_SCHEMA,
                mapping_profile=INCOME_STATEMENT_MAPPING_PROFILE,
                strict_equation_check=True,
            )


class TestFinancialReportSource:
    def test_source_uses_relevance_timestamp_and_exposes_diagnostics(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "balance.csv"
        _write_csv(
            csv_path,
            [
                {
                    "SECUCODE": "000001.SZ",
                    "REPORT_DATE": "2021-12-31 00:00:00",
                    "NOTICE_DATE": "2022-03-10 00:00:00",
                    "UPDATE_DATE": "",
                    "CURRENCY": "CNY",
                    "REPORT_TYPE": "年报",
                    "TOTAL_ASSETS": "100",
                    "TOTAL_LIABILITIES": "60",
                    "TOTAL_EQUITY": "40",
                },
                {
                    "SECUCODE": "000001.SZ",
                    "REPORT_DATE": "2021-12-31 00:00:00",
                    "NOTICE_DATE": "2022-03-20 00:00:00",
                    "UPDATE_DATE": "",
                    "CURRENCY": "CNY",
                    "REPORT_TYPE": "年报",
                    "TOTAL_ASSETS": "110",
                    "TOTAL_LIABILITIES": "70",
                    "TOTAL_EQUITY": "40",
                },
            ],
        )
        source = FinancialReportCSVSource(path=csv_path, kind="balance_sheet")
        scenario = Scenario()
        series = scenario.add_source(source)
        asyncio.run(scenario.run())

        assert len(series) == 1
        assert series.index[0] == ts_iso("2022-03-20 00:00:00")
        assert source.diagnostics.revision_dropped_rows == 1

    def test_scenario_integration_with_select_fields_and_coalesce(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "financial_reports" / "balance_raw_sample.csv"
        balance_source = FinancialReportCSVSource(path=fixture, kind="balance_sheet")
        other_source = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts_iso("2022-03-20 00:00:00")]),
            values=np.array([[1.0]], dtype=np.float64),
        )

        scenario = Scenario()
        balance_series = scenario.add_source(balance_source)
        other_series = scenario.add_source(other_source)

        assets_idx = balance_source.schema.field_index["balance_sheet.assets"]
        selector_series = scenario.add_operator(select(balance_series, (assets_idx,)))
        sum_series = scenario.add_operator(add(selector_series, other_series))
        asyncio.run(scenario.run())

        assert len(selector_series) == 1
        assert len(sum_series) == 1
        assert sum_series.index[0] == ts_iso("2022-03-20 00:00:00")
        assert float(sum_series.values[0][0]) == pytest.approx(101.0)


class TestGoldenFixtures:
    def test_balance_fixture_expected_values(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "financial_reports" / "balance_raw_sample.csv"
        source = FinancialReportCSVSource(path=fixture, kind="balance_sheet")
        scenario = Scenario()
        series = scenario.add_source(source)
        asyncio.run(scenario.run())

        i = source.schema.field_index
        row = series.values[0]
        assert float(row[i["balance_sheet.assets"]]) == pytest.approx(100.0)
        assert float(row[i["balance_sheet.liab"]]) == pytest.approx(-60.0)
        assert float(row[i["balance_sheet.equity"]]) == pytest.approx(-40.0)
        assert float(row[i["balance_sheet.assets.current.financial.fvpl.trading"]]) == pytest.approx(8.0)
        assert float(row[i["balance_sheet.assets.current.financial.fvpl.other"]]) == pytest.approx(12.0)
        assert float(row[i["balance_sheet.assets.current.other"]]) == pytest.approx(3.0)

    def test_income_fixture_expected_values(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "financial_reports" / "income_raw_sample.csv"
        source = FinancialReportCSVSource(path=fixture, kind="income_statement")
        scenario = Scenario()
        series = scenario.add_source(source)
        asyncio.run(scenario.run())

        i = source.schema.field_index
        row = series.values[0]
        assert float(row[i["income_statement.profit"]]) == pytest.approx(64.0)
        assert float(row[i["income_statement.profit.operating.income"]]) == pytest.approx(200.0)
        assert float(row[i["income_statement.profit.operating.expenses"]]) == pytest.approx(-120.0)
        assert float(row[i["income_statement.profit.other_income"]]) == pytest.approx(5.0)
        assert float(row[i["income_statement.profit.other_expenses"]]) == pytest.approx(-1.0)
        assert float(row[i["income_statement.profit.income_taxes"]]) == pytest.approx(-20.0)
