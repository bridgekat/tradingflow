//! Clock sources — emit `()` events at calendar-aligned timestamps.

use super::iter_source::IterSource;

/// Create a clock source from explicit timestamps (nanoseconds since epoch).
///
/// The output node holds `()` (zero-sized, purely a trigger).
pub fn clock(timestamps: Vec<i64>) -> IterSource<()> {
    IterSource::new(timestamps.into_iter().map(|ts| (ts, ())), ())
}

/// Generate daily timestamps (midnight in the given timezone) between
/// `start_ns` and `end_ns` (inclusive bounds, nanoseconds since epoch).
///
/// `tz`: IANA timezone name (e.g. `"Asia/Shanghai"`, `"US/Eastern"`).
///
/// # Panics
///
/// Panics if `tz` is not a valid IANA timezone name.
pub fn daily_clock(start_ns: i64, end_ns: i64, tz: &str) -> IterSource<()> {
    let timestamps = generate_calendar_timestamps(start_ns, end_ns, tz, CalendarFreq::Daily);
    clock(timestamps)
}

/// Generate monthly timestamps (midnight on the first day of each month
/// in the given timezone) between `start_ns` and `end_ns`.
///
/// # Panics
///
/// Panics if `tz` is not a valid IANA timezone name.
pub fn monthly_clock(start_ns: i64, end_ns: i64, tz: &str) -> IterSource<()> {
    let timestamps = generate_calendar_timestamps(start_ns, end_ns, tz, CalendarFreq::Monthly);
    clock(timestamps)
}

// ---------------------------------------------------------------------------
// Calendar timestamp generation
// ---------------------------------------------------------------------------

enum CalendarFreq {
    Daily,
    Monthly,
}

fn generate_calendar_timestamps(
    start_ns: i64,
    end_ns: i64,
    tz: &str,
    freq: CalendarFreq,
) -> Vec<i64> {
    use chrono::{DateTime, Datelike, TimeZone};
    use chrono_tz::Tz;

    let tz: Tz = tz
        .parse()
        .unwrap_or_else(|_| panic!("invalid timezone: {tz}"));

    let start_utc = DateTime::from_timestamp_nanos(start_ns);
    let end_utc = DateTime::from_timestamp_nanos(end_ns);

    let start_local = start_utc.with_timezone(&tz);
    let end_local = end_utc.with_timezone(&tz);

    let mut timestamps = Vec::new();
    let mut date = start_local.date_naive();
    let end_date = end_local.date_naive();

    // Align to frequency boundary.
    match freq {
        CalendarFreq::Monthly => {
            // Advance to the first day of the current or next month.
            if date.day() != 1 {
                date = if date.month() == 12 {
                    chrono::NaiveDate::from_ymd_opt(date.year() + 1, 1, 1).unwrap()
                } else {
                    chrono::NaiveDate::from_ymd_opt(date.year(), date.month() + 1, 1).unwrap()
                };
            }
        }
        CalendarFreq::Daily => {} // already aligned
    }

    while date <= end_date {
        // Convert local midnight to UTC nanoseconds.
        let midnight = date.and_hms_opt(0, 0, 0).unwrap();
        if let Some(utc) = tz.from_local_datetime(&midnight).earliest() {
            let ns = utc.timestamp_nanos_opt().unwrap();
            if ns >= start_ns && ns <= end_ns {
                timestamps.push(ns);
            }
        }

        // Advance.
        match freq {
            CalendarFreq::Daily => {
                date = date.succ_opt().unwrap();
            }
            CalendarFreq::Monthly => {
                date = if date.month() == 12 {
                    chrono::NaiveDate::from_ymd_opt(date.year() + 1, 1, 1).unwrap()
                } else {
                    chrono::NaiveDate::from_ymd_opt(date.year(), date.month() + 1, 1).unwrap()
                };
            }
        }
    }

    timestamps
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daily_clock_generates_days() {
        // 2024-01-01 00:00 UTC+8 to 2024-01-05 00:00 UTC+8
        let start = chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let end = chrono::NaiveDate::from_ymd_opt(2024, 1, 5)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();

        use chrono::TimeZone;
        let tz: chrono_tz::Tz = "Asia/Shanghai".parse().unwrap();
        let start_ns = tz
            .from_local_datetime(&start)
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap();
        let end_ns = tz
            .from_local_datetime(&end)
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap();

        let ts =
            generate_calendar_timestamps(start_ns, end_ns, "Asia/Shanghai", CalendarFreq::Daily);
        assert_eq!(ts.len(), 5); // Jan 1,2,3,4,5
    }

    #[test]
    fn monthly_clock_generates_months() {
        use chrono::TimeZone;
        let tz: chrono_tz::Tz = "Asia/Shanghai".parse().unwrap();

        let start = chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let end = chrono::NaiveDate::from_ymd_opt(2024, 6, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();

        let start_ns = tz
            .from_local_datetime(&start)
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap();
        let end_ns = tz
            .from_local_datetime(&end)
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap();

        let ts =
            generate_calendar_timestamps(start_ns, end_ns, "Asia/Shanghai", CalendarFreq::Monthly);
        assert_eq!(ts.len(), 6); // Jan, Feb, Mar, Apr, May, Jun
    }
}
