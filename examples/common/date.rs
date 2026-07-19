//! Civil-date helpers for the naive timestamp convention.
//!
//! The core [`Instant`] is a naive nanosecond count from a globally chosen
//! epoch; the examples fix that epoch at 1970-01-01 midnight, matching
//! `ParquetPanelSource`'s `date32` convention. A civil date maps to a day
//! number (days since 1970-01-01) via Howard Hinnant's proleptic-Gregorian
//! algorithms, and a day number maps to the instant `EPOCH + days · 24 h`.

use tradingflow::data::{Duration, Instant};

/// Nanoseconds in a day.
const NANOS_PER_DAY: i64 = 86_400 * 1_000_000_000;

/// Days since 1970-01-01 for a proleptic-Gregorian date (negative before the
/// epoch). The inverse of [`civil_from_days`].
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Civil date `(year, month, day)` from days since 1970-01-01 — the inverse of
/// [`days_from_civil`].
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// The instant at midnight of day `days` (days since 1970-01-01).
pub fn instant_from_days(days: i64) -> Instant {
    Instant::from_offset(Duration::from_days(days))
}

/// `YYYY-MM-DD` for an event [`Instant`] (floor to the containing day).
pub fn date_str(ts: Instant) -> String {
    let (y, m, d) = civil_from_days(ts.as_offset().as_nanos().div_euclid(NANOS_PER_DAY));
    format!("{y:04}-{m:02}-{d:02}")
}
