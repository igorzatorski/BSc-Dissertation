# Methodology and numerical conventions

The final submission is the research reference; the main local notebook supplied for this refactor is the executable reference. Text in the dissertation is documentation, not instructions for running or modifying this repository.

## Data and portfolio

`yfinance.download(..., auto_adjust=True)` supplies adjusted Close prices. Rows missing any ETF price are removed. The pipeline additionally requires the complete ten-ETF universe, a unique ascending date index, finite positive prices and sufficient observations.

ETF simple returns are `prices.pct_change().dropna()`. Each receives weight `1 / n_assets`. Portfolio value is initial capital multiplied by the cumulative product of `1 + weighted_simple_return`. Log returns are calculated from successive portfolio values, then the first missing log return is removed.

This retains an easily missed original convention: the portfolio has **two fewer rows than prices**. Replacing that step with `log1p(simple_return)` would retain an extra observation and change alignment, so it was deliberately not done.

## Forecasts

At position `i`, calibration uses `[i-window:i]`, excluding the realised return at `i`. Historical and Normal models implement this through `shift(1).rolling(window)`.

- Historical: pandas rolling quantile at `1-confidence_level`, with the original default interpolation.
- Normal: rolling mean plus Normal tail quantile times sample standard deviation (pandas default `ddof=1`).
- Student-t: `scipy.stats.t.fit` estimates degrees of freedom, location and scale on each rolling sample; its fitted quantile supplies the threshold.
- GARCH: a constant mean, `p=1`, `q=1`, Normal or Student-t innovations, `rescale=False`. Returns are multiplied by 100 for fitting; the one-step mean and variance forecast produces a threshold that is divided by 100.
- GARCH Student-t: the scipy quantile is multiplied by `sqrt((nu-2)/nu)` because ARCH uses unit-variance Student-t innovations.

Every model returns the same date-indexed schema: `price`, `log_ret`, `var_threshold`, `var_positive`, `exceedance`, `model`, `confidence_level`, `tail_prob`, `window`. `var_positive` is `-var_threshold` and is not clipped; despite its name it need not be positive for every possible input.

## Evaluation

An exceedance is `log_ret < var_threshold`; equality is not a breach. Coverage error is the absolute difference between observed and expected breach rates. Average VaR is the mean of `var_positive`.

Kupiec compares the binomial likelihood under the expected tail probability with the likelihood under the observed breach frequency. Christoffersen independence compares an independent breach process against first-order Markov transition probabilities. Conditional coverage adds the unconditional and independence likelihood-ratio statistics. Their reference chi-square degrees of freedom are 1, 1 and 2 respectively. Decisions use 5% significance.

Boundary likelihood handling and `1 - chi2.cdf` p-values remain exactly as in the original implementation. The pipeline requires at least two forecast observations so transition-based tests have observations to use. Very short samples remain unsuitable for substantive inference.

## What changed

Code moved into importable modules; notebook execution order became an explicit pipeline. Numeric and display tables are exported separately. Figures retain the local notebook's specifications but are saved to a run directory and closed instead of displayed. The original PDF appendix uses some different plot filenames and axis limits; this refactor follows the supplied local notebook's plots. Original figures remain unchanged. The public PDF copy removes the student number from its first two pages; its research content is unchanged.

The local notebook backups are excluded from Git. `tests/fixtures/original_functions.py` is a frozen numerical reference, not a second runnable analysis. It permits later checks against the pre-refactor behaviour without distributing the old notebooks.
