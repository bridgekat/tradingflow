"""Load equity structure and financial report data for an A-shares stock,
compute market cap and annualized financial metrics, and plot.

Requires `pip install -e ".[examples]"` and A-shares market data downloaded
via the crawler. See `python -m a_shares_crawler --help` for configuration
and download instructions.
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from a_shares_crawler.types import Schema as CSVSchema

from tradingflow import Scenario, Schema
from tradingflow.sources import CSVSource
from tradingflow.sources.stocks import FinancialReportSource
from tradingflow.operators import Map, Record, Select
from tradingflow.operators.num import Divide, Multiply, Negate
from tradingflow.operators.rolling import RollingMean
from tradingflow.operators.stocks import Annualize


PRICE_SCHEMA = Schema(CSVSchema.daily_prices().iter_field_ids())
EQUITY_SCHEMA = Schema(CSVSchema.equity_structures().iter_field_ids())
DIVIDEND_SCHEMA = Schema(CSVSchema.dividends().iter_field_ids())

BS_SCHEMA = Schema(CSVSchema.balance_sheet().iter_field_ids())
INC_SCHEMA = Schema(CSVSchema.income_statement().iter_field_ids())
CF_SCHEMA = Schema(CSVSchema.cash_flow_statement().iter_field_ids())


def build_scenario(symbol: str, data_dir: Path) -> tuple[Scenario, dict]:
    """Build a scenario with market cap and annualized financial metrics."""

    history_dir = data_dir / "a_shares_history"
    sc = Scenario()

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    # Daily prices. Shape: (6,).
    prices = sc.add_source(
        CSVSource(
            history_dir / f"{symbol}.daily_prices.csv",
            PRICE_SCHEMA,
            time_column="date",
        )
    )

    # Equity structure events. Shape: (2,).
    equity_structures = sc.add_source(
        CSVSource(
            history_dir / f"{symbol}.equity_structures.csv",
            EQUITY_SCHEMA,
            time_column="date",
        )
    )

    # Dividend events. Shape: (2,).
    dividends = sc.add_source(
        CSVSource(
            history_dir / f"{symbol}.dividends.csv",
            DIVIDEND_SCHEMA,
            time_column="date",
        )
    )

    # Balance sheet (quarterly, point-in-time). No annualization needed.
    balance = sc.add_source(
        FinancialReportSource(
            history_dir / f"{symbol}.balance_sheets.csv",
            BS_SCHEMA,
            report_date_column="date",
            notice_date_column="notice_date",
            use_effective_date=False,
        )
    )

    # Income statement (quarterly, YTD cumulative).
    income_ytd = sc.add_source(
        FinancialReportSource(
            history_dir / f"{symbol}.income_statements.csv",
            INC_SCHEMA,
            report_date_column="date",
            notice_date_column="notice_date",
            with_report_date=True,
            use_effective_date=False,
        )
    )

    # Cash flow statement (quarterly, YTD cumulative).
    cf_ytd = sc.add_source(
        FinancialReportSource(
            history_dir / f"{symbol}.cash_flow_statements.csv",
            CF_SCHEMA,
            report_date_column="date",
            notice_date_column="notice_date",
            with_report_date=True,
            use_effective_date=False,
        )
    )

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    # Annualize income and cash flow (strips year/day_of_year metadata).
    income_ann = sc.add_operator(Annualize(income_ytd))
    cf_ann = sc.add_operator(Annualize(cf_ytd))

    # Market cap = close price × total shares.
    close = sc.add_operator(Select(prices, PRICE_SCHEMA.index("prices.close")))
    total_shares = sc.add_operator(Select(equity_structures, EQUITY_SCHEMA.index("shares.total")))
    market_cap = sc.add_operator(Multiply(close, total_shares))

    # Balance sheet: total assets and equity.
    assets = sc.add_operator(Select(balance, BS_SCHEMA.index("balance_sheet.assets")))
    negative_equity = sc.add_operator(Select(balance, BS_SCHEMA.index("balance_sheet.equity")))
    equity = sc.add_operator(Negate(negative_equity))

    # Parent equity = sum of capital, reserves and parent interests.
    negative_parent_equity_components = sc.add_operator(
        Select(
            balance,
            BS_SCHEMA.indices(
                [
                    "balance_sheet.equity.capital",
                    "balance_sheet.equity.reserves",
                    "balance_sheet.equity.parent_interests",
                ]
            ),
        )
    )
    parent_equity = sc.add_operator(
        Map(
            negative_parent_equity_components,
            lambda x: -x.sum(),
            shape=(),
            dtype=np.float64,
        )
    )

    # Income statement (annualized): select income, expenses and net profit.
    op_income = sc.add_operator(Select(income_ann, INC_SCHEMA.index("income_statement.profit.operating.income")))
    net_profit = sc.add_operator(Select(income_ann, INC_SCHEMA.index("income_statement.profit")))

    # Cash flow (annualized): select operating, investing, financing.
    cash_flow = sc.add_operator(Select(cf_ann, CF_SCHEMA.index("cash_flow_statement.change")))

    # TTM net profit: rolling mean of annualized quarterly values over 365
    # days on the report-date axis. For equal-length quarters this equals
    # the sum of the last 4 quarterly values.
    net_profit_series = sc.add_operator(Record(net_profit))
    net_profit_ttm = sc.add_operator(RollingMean(net_profit_series, window=np.timedelta64(365, "D")))

    # Valuation ratios.
    ep_ratio = sc.add_operator(Divide(net_profit_ttm, market_cap))
    bp_ratio = sc.add_operator(Divide(parent_equity, market_cap))
    roe = sc.add_operator(Divide(net_profit_ttm, parent_equity))

    # Record into series for plotting.
    return sc, {
        "market_cap": sc.add_operator(Record(market_cap)),
        "assets": sc.add_operator(Record(assets)),
        "equity": sc.add_operator(Record(equity)),
        "parent_equity": sc.add_operator(Record(parent_equity)),
        "op_income": sc.add_operator(Record(op_income)),
        "net_profit": sc.add_operator(Record(net_profit)),
        "cash_flow": sc.add_operator(Record(cash_flow)),
        "ep_ratio": sc.add_operator(Record(ep_ratio)),
        "bp_ratio": sc.add_operator(Record(bp_ratio)),
        "roe": sc.add_operator(Record(roe)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", type=str, required=True, help='stock symbol (e.g. "000001.SZ")')
    parser.add_argument("--data-dir", type=Path, required=True, help="path to crawler data directory")
    args = parser.parse_args()

    symbol: str = args.symbol
    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(
            f"Data directory not found: {data_dir}\n"
            "Run `python -m a_shares_crawler --help` for download instructions."
        )

    # Run scenario.
    sc, handles = build_scenario(symbol, data_dir)
    sc.run()

    # Extract results.
    market_cap = sc.series_view(handles["market_cap"]).to_series()
    assets = sc.series_view(handles["assets"]).to_series()
    equity = sc.series_view(handles["equity"]).to_series()
    parent_equity = sc.series_view(handles["parent_equity"]).to_series()
    op_income = sc.series_view(handles["op_income"]).to_series()
    net_profit = sc.series_view(handles["net_profit"]).to_series()
    cash_flow = sc.series_view(handles["cash_flow"]).to_series()
    ep = sc.series_view(handles["ep_ratio"]).to_series()
    bp = sc.series_view(handles["bp_ratio"]).to_series()
    roe = sc.series_view(handles["roe"]).to_series()

    n = len(market_cap)
    if n == 0:
        raise SystemExit(f"No data found for {symbol}.")

    first = market_cap.index[0].date()
    last = market_cap.index[-1].date()
    print(f"{symbol}: {n} trading days, {first} to {last}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    plt.style.use(["fast"])
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Market cap, total assets, net assets (equity).
    ax = axes[0]
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.plot(assets.index, assets / 1e8, label="Total assets", color="C0", linewidth=0.8)
    ax.plot(equity.index, equity / 1e8, label="Net assets", color="C1", linewidth=0.8)
    ax.plot(
        parent_equity.index,
        parent_equity / 1e8,
        label="Parent equity",
        color="C1",
        linestyle="--",
        linewidth=0.8,
    )
    ax.plot(market_cap.index, market_cap / 1e8, label="Market cap", color="C2", linewidth=0.8)
    ax.set_ylabel("CNY (100M)")
    ax.set_title(f"{symbol} — Balance sheet & market cap")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 2: Annualized income statement, cash flows.
    ax = axes[1]
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.plot(op_income.index, op_income / 1e8, label="Operating income", color="C0", linewidth=0.8)
    ax.plot(net_profit.index, net_profit / 1e8, label="Net profit", color="C1", linewidth=0.8)
    ax.plot(cash_flow.index, cash_flow / 1e8, label="Cash flow", color="C2", linewidth=0.8)
    ax.set_ylabel("CNY (100M, annualized)")
    ax.set_title(f"{symbol} — Income & cash flow (annualized)")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 3: Valuation ratios.
    ax = axes[2]
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.plot(ep.index, ep * 100, label="E/P (TTM)", color="C0", linewidth=0.8)
    ax.plot(bp.index, bp * 100, label="B/P", color="C1", linewidth=0.8)
    ax.plot(roe.index, roe * 100, label="ROE (TTM)", color="C2", linewidth=0.8)
    ax.set_ylabel("%")
    ax.set_title(f"{symbol} — Valuation & profitability")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Date")

    fig.tight_layout()
    plt.show()
