# Design: columnar data storage for the flow backtester

Status: **decisions settled (2026-06-06)**; implementation pending. Author:
data-loading rework on `flowgraph-rebase`.
Scope: how the A-shares example data is stored and fed into the `flow` engine.

> **Guiding constraint (Zhanrong).** The store is a **pure format conversion** of
> the crawler's CSV output — reshape + re-encode only, **no feature derivation** —
> so changing strategy features later never requires re-crawling or re-ingesting.
> The conversion is an **optional post-step in
> [`a-shares-crawler`](https://github.com/bridgekat/a-shares-crawler)**:
> `--export-long {csv,parquet}...` picks the encoding(s); **both emit the same long
> schema**. TradingFlow only ever *reads*.

---

## 1. Problem

The example data is **75,752 CSV files, 4.1 GB**, one file per
`(symbol × data-kind)`: 5835 symbols × 13 kinds (7 logical + their `_raw` twins).
A cross-sectional run over the full universe pays, *every run*:

- **~29k source tasks → ~60–90k file opens.** `common::build_stacked` registers 5
  sources per symbol; each also runs `estimated_event_count` (an extra prefix read
  + two boundary seeks). On NTFS + Defender, open latency alone dominates.
- **Full-history rescans.** `CsvSource` streams from row 0 and filters at *emit*
  time. `daily_prices` runs 1991→present (~8369 rows/symbol); a 2022–2024 run still
  parses ~31 years per stock → **~49M `hifitime` date-parses per run**, uncached.
- **Wide-row waste.** `balance_sheets.csv` has **183 columns**; the strategy reads
  **3**. `income_statements` has 50, reads 1. Every field of every row is tokenized.
- **Text→f64 on every cell**, tens of millions of per-row `mpsc` `.await` hops
  merged across ~29k receivers, then re-stacked into cross-sections.
- **Nothing is cached** in binary form.

Two facts drive the design:

1. **The consumed numeric working set is small and fits in RAM.** A dense
   `[dates × stocks]` `f64` daily field is `8369 × 5835 × 8 B ≈ 390 MB`;
   close+volume ≈ 780 MB; quarterly fundamentals are megabytes.
2. **Consumption is cross-sectional, storage is symbol-major.** The engine is
   event-time driven and immediately `StackSync`s per-symbol series into one
   `[N_stocks]` vector *per date*. The on-disk layout is the **transpose** of what
   is consumed, rebuilt at runtime by merging ~29k timestamp-sorted streams.

---

## 2. Goals / non-goals

**Goals**

- One **sequential, columnar read** in event-time order — no per-symbol fan-in, no
  full-history rescan, no per-run date/number parsing.
- Read **only the fields and the time range** a run needs.
- A handful of files; a stable, documented long schema, identical across CSV and
  Parquet encodings.
- Friendly to ad-hoc analysis (pandas / polars / DuckDB read it directly).

**Non-goals**

- **No feature derivation in the store.** No forward-adjust, no TTM/annualization,
  no effective-date collapsing — all of that stays in the engine (see §6) so it
  remains changeable without touching the data.
- No live/streaming ingestion; the store is append-/rebuild-only.
- No change to strategy semantics or numerical results.

---

## 3. Format choice — the "DataFrame on disk" question

Yes, there are mature columnar formats that are the binary equivalent of a
DataFrame:

| Format | Columnar | Compress | Projection | Predicate pushdown | Zero-copy mmap | Notes |
|---|---|---|---|---|---|---|
| **Apache Parquet** | ✓ | ✓ | ✓ | ✓ (row-group min/max) | ✗ (decode) | De-facto standard; ~10× smaller than CSV; read by pandas/polars/DuckDB |
| **Arrow IPC / Feather v2** | ✓ | optional | ✓ | partial | ✓ | On-disk Arrow; `mmap` + slice, no decode |
| ORC / HDF5 / npy | ✓ | some | varies | varies | some | weaker Rust support / no stats |

**Decision: Parquet** for the binary path (projection + **date predicate
pushdown** + ~10× compression + universal tooling; row groups read sequentially
when the file is **date-sorted**). Decode lands in **Arrow** arrays. **Arrow
IPC/Feather** is the documented fallback if we later want pure `mmap`. The
crawler's `--export-long csv` emits the *same* long schema as text for
inspection/diffing; Parquet is the fast path.

Rust read path: **arrow-rs** (`arrow` + `parquet`) — streaming
`ParquetRecordBatchReader` with `ProjectionMask` + `RowFilter` on the date column.
(`polars` is an alternative if we later want an in-process DataFrame API.)

---

## 4. On-disk long schema (the crawler↔engine contract)

The crawler emits **one long table per data-kind, all symbols**, replacing the
per-symbol files. Same logical schema whether CSV or Parquet.

```
<data-dir>/
  symbol_list.csv                 # unchanged; TradingFlow's symbol↔index source
  daily_prices.parquet            # or .csv — long, date-sorted
  dividends.parquet
  equity_structures.parquet
  balance_sheets.parquet          # wide value-cols, report_date-sorted (see below)
  income_statements.parquet
  cash_flow_statements.parquet
  indirect_statements.parquet
```

**Regular daily kinds** (`daily_prices`, `dividends`, `equity_structures`) —
a tidy long panel: one row per `(date, symbol)`, the kind's existing value columns
kept as columns (already narrow, 2–7 cols; this is the truest "mere conversion"):

| column | type | notes |
|---|---|---|
| `date` | `date32` (days since 1970-01-01) | sort key; TradingFlow → TAI-ns via the existing `utc_to_tai(days*86_400e9)` / `instant_from_days` (UTC-midnight, matches `CsvSource` default) — **no per-row string parsing** |
| `symbol` | `string`, **dictionary-encoded** | mapped to the engine's dense index via `symbol_list.csv` |
| `prices.open/close/high/low/amount/volume` | `float64` | exactly the source CSV's columns |

Sorted by `(date, symbol)`. Row groups sized to a time block (≈ one quarter) so
the `date` range prunes whole groups.

**Statement kinds** (`balance_sheets` = 183 cols, etc.) — **same tidy shape, kept
wide**: one row per `(report_date, symbol)`, all line-items as columns. Parquet
column projection means a run needing 3 of 183 columns reads only those, so keeping
them wide costs nothing at read time and stays fully flexible (every field is
already present; a new feature needs no re-crawl):

| column | type | notes |
|---|---|---|
| `date` | `date32` | report period-end; **primary on-disk sort key** |
| `notice_date` | `date32`, nullable | publication date (may be missing); needed to compute the event time at read |
| `symbol` | `string` dict | |
| `error` | `bool` | crawler's balance-check flag (carried through verbatim) |
| `balance_sheet.equity.capital`, … (all line-items) | `float64` | the source CSV's columns verbatim |

Sorted by `(date, symbol)`. (The crawler names the period-end column `date`, matching
the per-symbol CSVs; this doc calls it the *report date* where the distinction from
`notice_date` matters.) **The effective-date logic (`max(date, notice_date)`,
retrospective-update dropping) is NOT applied here** — it stays in the engine (§5/§6),
so the look-ahead policy is tunable and the file carries no derived column. The on-disk
order therefore differs from the event order, which is why statement rows are
**reordered at read time** (§5).

Conventions:
- **Dates are `date32` (days)** — idiomatic Arrow, lossless, and converted by the
  exact arithmetic the examples already use.
- **Column names = the dotted names** the code already addresses by
  (`"prices.close"`, `"balance_sheet.equity.capital"`), so projection maps 1:1 with
  current column addressing.
- **All columns retained** (flexibility); projection picks what a run needs.
- **Symbol stays a string** dict — no separate id table; TradingFlow owns the
  symbol→index map (`load_symbols`).
- Drop the `_raw` twins (crawler's choice). No partitioning needed at this size; if
  it grows, partition `year=YYYY/`.

Why long, not wide-by-symbol: a `[date × symbol]` matrix has ~5835 columns and
breaks when the universe changes; long format keeps the schema stable, cheaply
dictionary-encodes the symbol, and streams one date-block at a time.

---

## 5. The backtest read path

Two new sources, mirroring the two access shapes (and the two existing source
types). Both turn one long table into **cross-sectional events**, collapsing the
~29k-stream fan-in into a single read.

**`ParquetPanelSource` — IMPLEMENTED** (`src/sources/parquet_panel.rs`). For the
regular date-keyed kinds. In the `init`-spawned (blocking) task:

1. Open the Parquet table; project `date`, `symbol`, and the requested value
   columns (so the 180-column statement tables don't decode unused fields). One
   forward pass over `(date, symbol)`-sorted row groups, no per-symbol seeks.
   (`with_time_range` filters emitted dates; row-group/page predicate pushdown is
   a TODO.)
2. Hold a running cross-section `Array<f64>` of shape `[N_symbols, R]` (`R` =
   value columns, `+2` leading `[year, day_of_year]` under `with_report_date`),
   indexed by the universe symbol order.
3. Accumulate the contiguous rows sharing a `date`, scatter each by `symbol`, then
   **emit one `(date, cross_section)` event per distinct date**.
4. **Pure StackSync — the carry lives downstream.** Each emitted cross-section
   reflects **only that date's rows** (absent symbols `NaN`); there is no
   carry-forward in the source and no window-start seeding — with a time range,
   rows before `start` are simply skipped. The "carry last value" / "NaN-fill"
   policy is the downstream `Stack` / `StackSync` operator's job, exactly as
   with the per-symbol sources. (The per-date NaN-reset is **sparse**: only the
   symbol rows written this date are cleared at the next date, so the irregular
   kinds stay O(rows), not O(N·K).)

Trait mapping (`src/source.rs`): `Event = Output = Array<f64>`; `write` assigns the
payload (like `CsvSource::write`). The per-stock pipeline is recovered downstream
by `Select::new(vec![i], 0, true)` (stock `i`'s `[R]` row) + a NaN `Filter` (so
message-passing operators see each real event once) — see §6.

**`ReportPanelSource` — IMPLEMENTED** (`src/sources/report_panel.rs`). For the
statement kinds, which have two date columns (`date` = period-end, `notice_date`)
and need a *computed* event order. Same StackSync emission as above, but
**load-and-sort** (the statement tables are tiny, ~690k rows) so it can reorder:

- `use_effective_date = false` (default) — events fire on the report `date`
  (equivalent to a `ParquetPanelSource` on `date`).
- `use_effective_date = true` (**what the strategy examples use**) — events fire on
  the *effective date* `e = max(date, notice_date)` (fallback when `notice_date` is
  null), so a report is not visible until published (no look-ahead). Rows are
  reordered by `e`, and **retrospective updates are dropped** per symbol (keep a
  report iff its `date` advances that symbol's running max). The reorder is wholly
  inside the source, so the engine sees a clean non-decreasing stream.

`with_report_date` prepends `[year, day_of_year]` of the report date via the **same
hifitime path** as `FinancialReportSource`, so `Annualize` matches it bit-for-bit.

**Validation.** A differential check builds the panel `build_stacked` and an inline
`CsvSource`/`FinancialReportSource` `build_stacked` (both with the same new
semantics — no start-carry, `use_effective_date=true`) and compares the stacked
panels: `close`, `volume`, `adjusted_close`, `adjusts`, `total_shares`,
`circ_shares`, **and `net_profit` match bit-exactly**; only `parent_equity` differs,
by `max_rel ≈ 2e-13` — pandas(parquet)-vs-Rust(CSV) float-parse noise amplified by
the 3-term `-(capital+reserves+parent_interests)` sum, far below any rank threshold.
`examples/plot_daily_price` / `plot_financial_data` were likewise checked
series-equal against the per-symbol pipelines.

**Event-count impact.** `daily_prices` drops from ~5835 streams × ~750 windowed
rows ≈ **4.4 M row-events** to ≈ **750 cross-section events** for a 3-year window —
removing most channel/merge/`stabilize` overhead on top of the I/O and parse wins.

---

## 6. Engine integration — `Select`-from-panel (chosen for now)

The per-symbol transforms `build_stacked` runs *before* stacking (`ForwardAdjust`,
`Annualize`, the balance `Map`-sum, the field `Select`s) are **kept unchanged**.
Instead of rewriting them cross-sectionally, each stock is recovered from the
panel with `Select::new(vec![i], 0, true)` — stock `i`'s `[R]` row, a drop-in for
the old per-symbol source output — and the existing operators run per stock, then
`StackSync`/`Stack` recombine into `[N]`. The fan-out becomes **`1 → N → 1`**: one
`ParquetPanelSource` → N per-stock branches → one stack. High node count, but the
~6000 *file opens* collapse to one sequential scan, which is the bottleneck. (A
later optimization could vectorize `ForwardAdjust`/`Annualize` over `[N]` to drop
the fan-out; deferred — "OK for now".)

**Irregular kinds need a NaN `Filter`.** The panel ticks on the *union* of all
stocks' event dates, so a per-stock `Select` fires on that union cadence with NaN
where the stock had no row. Feeding that into a message-passing operator
(`ForwardAdjust` treats each dividend tick as a new event) would double-count. So
each per-stock branch is `Filter(|row| row.iter().any(is_finite))`, which drops the
all-NaN "no data" cross-sections and leaves exactly that stock's real events —
verified identical to the per-symbol stream (§5). The carry-forward of state
(shares, equity) and NaN-fill of prices is then the existing `Stack` / `StackSync`'s
job, unchanged.

`CsvSource`/`FinancialReportSource` (whose **window-start carry was removed** — they
now drop everything before `start` rather than emitting the last pre-`start` value
at `start`, matching the panel sources) and the legacy per-symbol `build_stacked`
stay for single-series/ad-hoc use; the panel path is added alongside. The strategy
examples now set **`use_effective_date = true`** (the old per-symbol `build_stacked`
used report-date alignment, i.e. look-ahead-biased — now fixed).

---

## 7. Conversion step (in a-shares-crawler — **implemented**)

**Done.** Implemented in the vendored submodule `extern/a-shares-crawler` as
`a_shares_crawler/export.py` plus an `--export-long {csv,parquet}...` flag on the
crawler CLI (one or more formats at once). After the per-symbol download (or
standalone, on an existing download), it **reshapes and concatenates** the
per-symbol CSVs into the §4 long tables:

```sh
# as a post-step of a normal crawl:
python -m a_shares_crawler --config config.json --data-dir DIR --export-long csv parquet
# or standalone, no network/config (pure conversion of an existing download):
python -m a_shares_crawler.export --data-dir DIR --export-long csv parquet
```

For every kind it globs `<data-dir>/a_shares_history/*.<kind>.csv`, reads each,
tags it with the `symbol` parsed from the file name, concatenates, sorts by
`(date, symbol)`, and writes `<data-dir>/<kind>.{csv,parquet}`. Parquet casts
`date`/`notice_date` → `date32`, dictionary-encodes `symbol`, and uses
date-contiguous row groups; CSV carries the identical schema as text. The wide
statement columns are kept (no unpivot); nothing is derived, adjusted, or dropped.
`pyarrow` is a core crawler dependency (CSV export needs only pandas).
Embarrassingly parallel; run once, re-run when the crawler appends data.

Validated on a 3-symbol subset of the bundled data: `date32` dates + dictionary
`symbol` + nullable `notice_date`, statements kept wide (`error` + all line-items),
output date-monotonic and `(date, symbol)`-sorted, row counts and a spot-checked
value identical to the source CSVs, CSV/Parquet parity.

---

## 8. Migration status (TradingFlow side)

1. ✅ **Schema frozen** as the crawler↔engine contract (§4).
2. ✅ **arrow-rs read deps** added (non-optional); **`ParquetPanelSource`** +
   **`ReportPanelSource`** implemented (§5).
3. ✅ **`build_stacked` migrated** to the long-table panel path with per-stock
   `Select` + NaN `Filter` (§6), adapting all cross-sectional strategy examples;
   `plot_daily_price` / `plot_financial_data` rewritten to the panels too.
   Validated against the per-symbol pipelines (§5 *Validation*). `CsvSource` /
   `FinancialReportSource` remain for single-series/ad-hoc use.
4. **Deferred (optional):** cross-sectional `ForwardAdjust`/`Annualize` to drop the
   `1 → N → 1` fan-out; row-group date predicate pushdown (§5); a `--threads`
   benchmark of the end-to-end speedup (§9).

Additive to the `flow` source layer; no legacy-engine cutover involved.

---

## 9. Benchmarks to validate (define before building)

- **Cold load**: time-to-first-`stabilize` and total source-drain time,
  `CsvSource` vs `PanelSource`, at `--index-size {30, 1000, full}`. Target: data
  load from tens of seconds → low seconds.
- **Peak RAM** of the loaded panels (expect < 3 GB for the full daily set).
- **End-to-end** `mean_variance_strategy` wall-clock and the **serial-vs-`--threads
  8` speedup**: shrinking the largely-serial load raises the parallel fraction, so
  the 8-way cvxpy-solve parallelism should approach its ceiling (the original 1.5×
  motivation).
- **On-disk size** CSV vs Parquet (expect ~10×).
- **Numerical parity** on a few strategy outputs.

---

## 10. Settled decisions

| # | Decision | Choice |
|---|---|---|
| 1 | **Where conversion runs** | In **a-shares-crawler**, optional `--export-long {csv,parquet}...` (one or more formats); both emit the §4 long schema. TradingFlow reads only. |
| 2 | **Preprocessing** | **None (B2).** Pure reshape + re-encode; all transforms — incl. forward-adjust, annualize, and the financial effective-date/retrospective logic — stay in the engine, so features stay changeable. |
| 3 | **Table shape** | One table **per kind, all symbols**, same `(dates…, symbol, value-cols…)` shape for both. Daily kinds = `(date, symbol, value-cols…)`. Statements = `(report_date, notice_date, symbol, all-line-item cols…)` **kept wide** — Parquet projection reads only the needed columns, so width is free and maximally flexible. **No melting.** |
| 4 | **Dates** | `date32` (days since epoch); engine → TAI-ns via the existing UTC-midnight conversion. No per-row string parsing. |
| 5 | **Value / symbol types** | value columns `float64`; `symbol` **dictionary-encoded string**; symbol→index owned by the engine via `symbol_list.csv` (no id table). |
| 6 | **Column naming** | Columns = the existing dotted names (`prices.close`, `balance_sheet.equity.capital`), so projection maps 1:1 to current column addressing. Keep **all** columns. |
| 7 | **Read granularity + ordering** | Daily kinds (date == event order on disk) → streaming `PanelSource`, one sequential scan, emits a cross-section per date. Statements (on-disk `report_date` ≠ event order) → `FundamentalsSource`, which **reorders by effective date `e = max(report, notice)` in-source** (load-and-sort; or a `report_date`-watermarked min-heap since `e ≥ report_date`), applies per-symbol retrospective-drop, emits carry-forward cross-sections. |
| 8 | **Format / library** | Parquet on disk (Arrow in memory; Feather/`mmap` fallback). Rust read via **arrow-rs** (`arrow`/`parquet`). |
| 9 | **Compatibility** | Keep `CsvSource`/`FinancialReportSource` + legacy per-symbol `build_stacked`; add the long-table path and switch examples once long data exists. |

**Main implementation cost identified:** the cross-sectional `ForwardAdjust` and
`Annualize` operators (§6) — every other pre-stack transform is already
elementwise. The financial **read-time reorder** (decision 7) is the other piece of
real logic, but it is just the existing `FinancialReportSource` sort/retrospective
semantics, contained inside `FundamentalsSource`.
