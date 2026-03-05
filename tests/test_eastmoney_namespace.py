"""Tests for EastMoney source namespace wiring."""

from __future__ import annotations

import importlib

import src
import src.sources as sources
from src.sources.eastmoney import history as eastmoney_history
from src.sources.eastmoney.history import DailyMarketSnapshotCSVSource, FinancialReportCSVSource


def test_eastmoney_symbols_are_available_only_via_namespace() -> None:
    assert eastmoney_history.DailyMarketSnapshotCSVSource is DailyMarketSnapshotCSVSource
    assert eastmoney_history.FinancialReportCSVSource is FinancialReportCSVSource


def test_eastmoney_symbols_are_not_top_level_re_exports() -> None:
    assert not hasattr(src, "DailyMarketSnapshotCSVSource")
    assert not hasattr(src, "FinancialReportCSVSource")
    assert not hasattr(sources, "DailyMarketSnapshotCSVSource")
    assert not hasattr(sources, "FinancialReportCSVSource")


def test_legacy_adapter_modules_removed() -> None:
    try:
        importlib.import_module("src.sources.daily_market_snapshot")
        loaded_daily = True
    except ModuleNotFoundError:
        loaded_daily = False
    assert loaded_daily is False

    try:
        importlib.import_module("src.sources.financial_reports")
        loaded_financial = True
    except ModuleNotFoundError:
        loaded_financial = False
    assert loaded_financial is False
