//! Shared trading model for the native MODE and MAP-Elites example.
//!
//! An EMA/SMA crossing strategy buys or sells the whole position after
//! configurable waiting periods. Returns are divided by buy-and-hold returns
//! so stocks with very different absolute growth remain comparable.

use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use yahoo::time::{Date, Month, OffsetDateTime, PrimitiveDateTime, Time};
use yahoo_finance_api as yahoo;

pub const DIM: usize = 4;
pub const START_CASH: f64 = 1_000_000.0;
pub const LOWER: [f64; DIM] = [20.0, 50.0, 10.0, 10.0];
pub const UPPER: [f64; DIM] = [50.0, 100.0, 200.0, 200.0];
pub const QD_LOWER: [f64; DIM] = [0.4; DIM];
pub const QD_UPPER: [f64; DIM] = [1.6; DIM];

#[derive(Clone, Debug)]
pub struct PriceSeries {
    pub ticker: String,
    pub timestamps: Vec<i64>,
    pub close: Vec<f64>,
}

impl PriceSeries {
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.close.len() < 2 || self.timestamps.len() != self.close.len() {
            return Err(format!("{} has insufficient or inconsistent history", self.ticker).into());
        }
        if self
            .close
            .iter()
            .any(|price| !price.is_finite() || *price <= 0.0)
        {
            return Err(format!("{} contains an invalid adjusted close", self.ticker).into());
        }
        if self.timestamps.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(format!("{} timestamps are not strictly increasing", self.ticker).into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TradingEvaluation {
    pub factors: Vec<f64>,
    pub trades: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct TradingProblem {
    series: Vec<PriceSeries>,
    hodl: Vec<f64>,
    max_trades: usize,
}

impl TradingProblem {
    pub fn new(series: Vec<PriceSeries>, max_trades: usize) -> Result<Self, Box<dyn Error>> {
        if series.is_empty() {
            return Err("at least one ticker is required".into());
        }
        for values in &series {
            values.validate()?;
        }
        let hodl = series
            .iter()
            .map(|values| hodl(&values.close, START_CASH))
            .collect();
        Ok(Self {
            series,
            hodl,
            max_trades,
        })
    }

    pub fn tickers(&self) -> Vec<&str> {
        self.series
            .iter()
            .map(|values| values.ticker.as_str())
            .collect()
    }

    pub fn lengths(&self) -> Vec<usize> {
        self.series
            .iter()
            .map(|values| values.close.len())
            .collect()
    }

    pub fn hodl_factors(&self) -> &[f64] {
        &self.hodl
    }

    pub fn nobj(&self) -> usize {
        self.series.len()
    }

    pub fn max_trades(&self) -> usize {
        self.max_trades
    }

    pub fn evaluate(&self, x: &[f64]) -> TradingEvaluation {
        let [ema_period, sma_period, wait_buy, wait_sell] = parameters(x);
        let mut factors = Vec::with_capacity(self.series.len());
        let mut trades = Vec::with_capacity(self.series.len());
        for (values, &hodl_factor) in self.series.iter().zip(&self.hodl) {
            let (gain, count) = strategy(
                &values.close,
                START_CASH,
                ema_period,
                sma_period,
                wait_buy,
                wait_sell,
            );
            factors.push(gain / hodl_factor);
            trades.push(count);
        }
        TradingEvaluation { factors, trades }
    }

    /// MODE values: one minimized negative return per stock followed by one
    /// `trades - max_trades <= 0` constraint per stock.
    pub fn mo_values(&self, x: &[f64]) -> Vec<f64> {
        let evaluation = self.evaluate(x);
        let mut values = Vec::with_capacity(2 * self.nobj());
        values.extend(evaluation.factors.iter().map(|factor| -*factor));
        values.extend(
            evaluation
                .trades
                .iter()
                .map(|&trades| trades as f64 - self.max_trades as f64),
        );
        values
    }

    /// MAP-Elites minimizes the negative balanced return and uses the four
    /// per-stock relative returns as behavior descriptors.
    pub fn qd_value(&self, x: &[f64]) -> (f64, Vec<f64>) {
        let evaluation = self.evaluate(x);
        (-geometric_mean(&evaluation.factors), evaluation.factors)
    }
}

pub fn parameters(x: &[f64]) -> [usize; DIM] {
    assert_eq!(
        x.len(),
        DIM,
        "trading decision vector must have four values"
    );
    [
        x[0].clamp(LOWER[0], UPPER[0]) as usize,
        x[1].clamp(LOWER[1], UPPER[1]) as usize,
        x[2].clamp(LOWER[2], UPPER[2]) as usize,
        x[3].clamp(LOWER[3], UPPER[3]) as usize,
    ]
}

pub fn hodl(close: &[f64], start_cash: f64) -> f64 {
    let shares = (start_cash / close[0]) as u64;
    let cash = start_cash - shares as f64 * close[0];
    (cash + shares as f64 * close[close.len() - 1]) / start_cash
}

/// Run the EMA/SMA crossing strategy, returning absolute cash gain and the
/// number of signal-triggered trades. Liquidation at the final close is not
/// counted as a signal, matching the Python example.
pub fn strategy(
    close: &[f64],
    start_cash: f64,
    ema_period: usize,
    sma_period: usize,
    wait_buy: usize,
    wait_sell: usize,
) -> (f64, usize) {
    assert!(!close.is_empty());
    let ema_period = ema_period.max(1);
    let sma_period = sma_period.max(1);
    let alpha = 2.0 / (ema_period as f64 + 1.0);
    let mut ema = close[0];
    let mut rolling_sum = 0.0;
    let mut cash = start_cash;
    let mut shares = 0u64;
    let mut last_trade = 0usize;
    let mut num_trades = 0usize;

    for (i, &price) in close.iter().enumerate() {
        if i > 0 {
            ema = alpha * price + (1.0 - alpha) * ema;
        }
        rolling_sum += price;
        if i >= sma_period {
            rolling_sum -= close[i - sma_period];
        }
        if i + 1 < ema_period || i + 1 < sma_period {
            continue;
        }
        let sma = rolling_sum / sma_period as f64;
        if shares == 0 && ema > sma && i > last_trade.saturating_add(wait_buy) {
            let bought = (cash / price) as u64;
            cash -= bought as f64 * price;
            shares += bought;
            last_trade = i;
            num_trades += 1;
        } else if shares > 0 && ema < sma && i > last_trade.saturating_add(wait_sell) {
            cash += shares as f64 * price;
            shares = 0;
            last_trade = i;
            num_trades += 1;
        }
    }
    cash += shares as f64 * close[close.len() - 1];
    (cash / start_cash, num_trades)
}

pub fn geometric_mean(values: &[f64]) -> f64 {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return 0.0;
    }
    values
        .iter()
        .product::<f64>()
        .powf(1.0 / values.len() as f64)
}

/// Best balanced relative return represented by a feasible Pareto front.
pub fn pareto_quality(front: &[Vec<f64>], nobj: usize) -> f64 {
    front
        .iter()
        .filter(|values| values.len() >= nobj)
        .map(|values| {
            let factors: Vec<f64> = values[..nobj].iter().map(|value| -*value).collect();
            geometric_mean(&factors)
        })
        .fold(0.0, f64::max)
}

/// Capacity-normalized sum of balanced returns in an archive. Empty niches
/// contribute zero, so this rewards both coverage and solution quality.
pub fn archive_quality(ys: &[f64]) -> f64 {
    if ys.is_empty() {
        return 0.0;
    }
    ys.iter()
        .filter(|value| value.is_finite() && **value < 0.0)
        .map(|value| -*value)
        .sum::<f64>()
        / ys.len() as f64
}

pub async fn load_histories(
    tickers: &[String],
    start: &str,
    end: &str,
    cache_dir: &Path,
    offline: bool,
) -> Result<Vec<PriceSeries>, Box<dyn Error>> {
    let start_time = parse_date(start)?;
    let end_time = parse_date(end)?;
    if start_time >= end_time {
        return Err("start date must be before end date".into());
    }
    fs::create_dir_all(cache_dir)?;
    let mut histories = Vec::with_capacity(tickers.len());
    let provider = yahoo::YahooConnector::new()?;
    for ticker in tickers {
        let path = cache_path(cache_dir, ticker, start, end);
        let values = if path.exists() {
            read_cache(&path, ticker)?
        } else if offline {
            return Err(format!("missing offline cache {}", path.display()).into());
        } else {
            let response = provider
                .get_quote_history_interval(ticker, start_time, end_time, "1d")
                .await?;
            let mut quotes = response.quotes()?;
            quotes.sort_by_key(|quote| quote.timestamp);
            let values = PriceSeries {
                ticker: ticker.clone(),
                timestamps: quotes
                    .iter()
                    .filter(|quote| quote.timestamp >= start_time.unix_timestamp())
                    .filter(|quote| quote.timestamp < end_time.unix_timestamp())
                    .map(|quote| quote.timestamp)
                    .collect(),
                close: quotes
                    .iter()
                    .filter(|quote| quote.timestamp >= start_time.unix_timestamp())
                    .filter(|quote| quote.timestamp < end_time.unix_timestamp())
                    .map(|quote| quote.adjclose)
                    .collect(),
            };
            values.validate()?;
            write_cache(&path, &values)?;
            values
        };
        values.validate()?;
        histories.push(values);
    }
    Ok(histories)
}

fn parse_date(value: &str) -> Result<OffsetDateTime, Box<dyn Error>> {
    let mut fields = value.split('-');
    let year: i32 = fields.next().ok_or("missing year")?.parse()?;
    let month: u8 = fields.next().ok_or("missing month")?.parse()?;
    let day: u8 = fields.next().ok_or("missing day")?.parse()?;
    if fields.next().is_some() {
        return Err("dates must use YYYY-MM-DD".into());
    }
    let date = Date::from_calendar_date(year, Month::try_from(month)?, day)?;
    Ok(PrimitiveDateTime::new(date, Time::MIDNIGHT).assume_utc())
}

fn cache_path(cache_dir: &Path, ticker: &str, start: &str, end: &str) -> PathBuf {
    let ticker = ticker.replace(['/', '\\', ':'], "_");
    cache_dir.join(format!("trading_{ticker}_{start}_{end}.csv"))
}

fn read_cache(path: &Path, ticker: &str) -> Result<PriceSeries, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut timestamps = Vec::new();
    let mut close = Vec::new();
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line_number == 0 && line.trim() == "timestamp,close" {
            continue;
        }
        let (timestamp, price) = line
            .split_once(',')
            .ok_or_else(|| format!("malformed cache row {}", line_number + 1))?;
        timestamps.push(timestamp.parse()?);
        close.push(price.parse()?);
    }
    Ok(PriceSeries {
        ticker: ticker.to_string(),
        timestamps,
        close,
    })
}

fn write_cache(path: &Path, values: &PriceSeries) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "timestamp,close")?;
    for (&timestamp, &price) in values.timestamps.iter().zip(&values.close) {
        writeln!(writer, "{timestamp},{price:.17}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prices() -> Vec<f64> {
        vec![10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 13.0, 12.0]
    }

    #[test]
    fn hodl_preserves_cash_remainder() {
        let factor = hodl(&[3.0, 6.0], 10.0);
        assert!((factor - 1.9).abs() < 1e-12);
    }

    #[test]
    fn strategy_is_finite_and_counts_signals() {
        let (factor, trades) = strategy(&prices(), 100.0, 2, 3, 0, 0);
        assert!(factor.is_finite() && factor > 0.0);
        assert!(trades > 0);
    }

    #[test]
    fn problem_shapes_and_quality_metrics() {
        let series = vec![PriceSeries {
            ticker: "TEST".to_string(),
            timestamps: (1..=prices().len() as i64).collect(),
            close: prices(),
        }];
        let problem = TradingProblem::new(series, 3).unwrap();
        let x = [20.0, 50.0, 10.0, 10.0];
        let mo = problem.mo_values(&x);
        assert_eq!(mo.len(), 2);
        let (y, descriptor) = problem.qd_value(&x);
        assert_eq!(descriptor.len(), 1);
        assert!((y + descriptor[0]).abs() < 1e-12);
        assert_eq!(pareto_quality(&[mo], 1), descriptor[0]);
        assert_eq!(archive_quality(&[-2.0, f64::INFINITY]), 1.0);
    }

    #[test]
    fn dates_and_parameters_are_validated_and_clamped() {
        assert_eq!(
            parse_date("2020-01-02").unwrap().unix_timestamp(),
            1_577_923_200
        );
        assert!(parse_date("bad").is_err());
        assert_eq!(parameters(&[1.0, 500.0, 15.9, 20.1]), [20, 100, 15, 20]);
    }
}
