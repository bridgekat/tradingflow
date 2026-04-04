"""Load equity structure and financial report data for an A-shares stock,
compute market cap and annualized financial metrics, and plot.

Requires `pip install -e ".[examples]"` and A-shares market data downloaded
via the crawler. See `python -m a_shares_crawler --help` for configuration
and download instructions.
"""

from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from a_shares_crawler.types import Schema as CrawlerSchema

from tradingflow import Scenario, Schema
from tradingflow.sources import CSVSource
from tradingflow.sources.stocks import FinancialReportSource
from tradingflow.operators import Record, Select
from tradingflow.operators.num import Multiply
from tradingflow.operators.stocks import Annualize


PRICE_SCHEMA = Schema(["open", "close", "high", "low", "amount", "volume"])
DIVIDEND_SCHEMA = Schema(["share_dividends", "cash_dividends"])
EQUITY_SCHEMA = Schema(["total_shares", "circulating_shares"])

BS_SCHEMA = Schema(CrawlerSchema.balance_sheet().iter_field_ids())
INC_SCHEMA = Schema(CrawlerSchema.income_statement().iter_field_ids())
CF_SCHEMA = Schema(CrawlerSchema.cash_flow_statement().iter_field_ids())


def build_scenario(symbol: str, data_dir: Path) -> tuple[Scenario, dict]:
    """Build a scenario with market cap and annualized financial metrics."""
    history_dir = data_dir / "a_shares_history"
    sc = Scenario()

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    # Daily prices. Shape: (6,).
    prices = sc.add_source(CSVSource(history_dir / f"{symbol}.daily_prices.csv", PRICE_SCHEMA))

    # Dividend events. Shape: (2,).
    dividends = sc.add_source(CSVSource(history_dir / f"{symbol}.dividends.csv", DIVIDEND_SCHEMA))

    # Equity structure (irregular updates). Shape: (2,).
    # Uses FinancialReportSource for correct two-timestamp handling.
    equity = sc.add_source(
        FinancialReportSource(
            history_dir / f"{symbol}.equity_structures.csv",
            EQUITY_SCHEMA,
            report_date_column="date",
            notice_date_column="notice_date",
            use_effective_date=False,
        )
    )

    # Balance sheet (quarterly, point-in-time). No annualization needed.
    balance = sc.add_source(
        FinancialReportSource(
            history_dir / f"{symbol}.balance_sheets.csv",
            BS_SCHEMA,
            report_date_column="report_date",
            notice_date_column="notice_date",
            use_effective_date=False,
        )
    )

    # Income statement (quarterly, YTD cumulative).
    income_ytd = sc.add_source(
        FinancialReportSource(
            history_dir / f"{symbol}.income_statements.csv",
            INC_SCHEMA,
            report_date_column="report_date",
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
            report_date_column="report_date",
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
    close = sc.add_operator(Select(prices, [PRICE_SCHEMA.index("close")]))
    total_shares = sc.add_operator(Select(equity, [EQUITY_SCHEMA.index("total_shares")]))
    market_cap = sc.add_operator(Multiply(close, total_shares))

    # Balance sheet: select total assets and negative equity together.
    bs_metrics = sc.add_operator(
        Select(
            balance,
            BS_SCHEMA.indices(["balance_sheet.assets", "balance_sheet.equity"]),
        )
    )

    # Income statement (annualized): select income, expenses and net profit.
    inc_metrics = sc.add_operator(
        Select(
            income_ann,
            INC_SCHEMA.indices(["income_statement.profit.operating.income", "income_statement.profit"]),
        )
    )

    # Cash flow (annualized): select operating, investing, financing.
    cf_metrics = sc.add_operator(
        Select(
            cf_ann,
            CF_SCHEMA.indices(["cash_flow_statement.change"]),
        )
    )

    # Record into series for plotting.
    return sc, {
        "market_cap": sc.add_operator(Record(market_cap)),
        "balance_sheet": sc.add_operator(Record(bs_metrics)),
        "income": sc.add_operator(Record(inc_metrics)),
        "cash_flow": sc.add_operator(Record(cf_metrics)),
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

    sc, handles = build_scenario(symbol, data_dir)
    sc.run()

    # Extract series as DataFrames.
    market_cap_df = sc.series_view(handles["market_cap"]).to_dataframe(["market_cap"])
    bs_df = sc.series_view(handles["balance_sheet"]).to_dataframe(["total_assets", "negative_equity"])
    inc_df = sc.series_view(handles["income"]).to_dataframe(["operating_income", "net_profit"])
    cf_df = sc.series_view(handles["cash_flow"]).to_dataframe(["change"])

    n = len(market_cap_df)
    if n == 0:
        raise SystemExit(f"No data found for {symbol}.")

    first = market_cap_df.index[0].date()
    last = market_cap_df.index[-1].date()
    print(f"{symbol}: {n} trading days, {first} to {last}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Market cap, total assets, net assets (equity).
    ax = axes[0]
    ax.plot(bs_df.index, bs_df["total_assets"] / 1e8, "o-", label="Total assets", markersize=3)
    ax.plot(bs_df.index, -bs_df["negative_equity"] / 1e8, "s-", label="Net assets", markersize=3)
    ax.plot(market_cap_df.index, market_cap_df["market_cap"] / 1e8, label="Market cap", linewidth=0.8)
    ax.set_ylabel("CNY (100M)")
    ax.set_title(f"{symbol} — Balance sheet & market cap")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 2: Annualized income statement, cash flows.
    ax = axes[1]
    ax.plot(inc_df.index, inc_df["operating_income"] / 1e8, "o-", label="Operating income", markersize=3)
    ax.plot(inc_df.index, inc_df["net_profit"] / 1e8, "^-", label="Net profit", markersize=3)
    ax.plot(cf_df.index, cf_df["change"] / 1e8, "o-", label="Cash flow", markersize=3)
    ax.set_ylabel("CNY (100M, annualized)")
    ax.set_title(f"{symbol} — Income & cash flow (annualized)")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    plt.show()
