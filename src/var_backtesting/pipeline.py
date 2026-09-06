"""Explicit download -> portfolio -> forecasts -> backtests -> report workflow."""

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .backtesting import summarise_all_backtests
from .data import load_prices, validate_prices
from .models import MODELS
from .portfolio import build_equal_weighted_portfolio
from .reporting import create_clean_backtesting_table


def run_analysis(prices, output_dir, window=config.window, models=None, plots=True):
    """Write a complete run to a new directory; never overwrite a previous run."""
    prices = validate_prices(prices, config.tickers)
    portfolio = build_equal_weighted_portfolio(prices, config.initial_value)
    if window < 2 or len(portfolio) < window + 2:
        raise ValueError("Need at least window + 2 portfolio returns for backtesting.")
    model_names = list(MODELS) if models is None else list(models)
    if not model_names or any(name not in MODELS for name in model_names):
        raise ValueError("Select at least one supported model.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "forecasts").mkdir()
    prices.to_csv(output_dir / "prices.csv", index_label="Date")
    portfolio.to_csv(output_dir / "portfolio.csv", index_label="Date")
    config.etf_info.to_csv(output_dir / "etf_universe.csv", index=False)
    results = {}
    convergence_issues = []
    started = time.perf_counter()
    for name in model_names:
        for level in config.confidence_levels:
            print(
                f"Estimating {name} at {level:.0%} ({len(portfolio)-window:,} forecasts)...",
                flush=True,
            )
            result = MODELS[name](portfolio, window, level)
            if (
                len(result) != len(portfolio) - window
                or not np.isfinite(result["var_threshold"]).all()
            ):
                raise RuntimeError(
                    f"Incomplete/nonfinite forecasts for {name} at {level}."
                )
            for issue in result.attrs.get("convergence_issues", []):
                convergence_issues.append(
                    {"model": name, "confidence_level": level, **issue}
                )
            results[name, level] = result
            result.to_csv(
                output_dir / "forecasts" / f"{name}_{level:.0%}.csv", index_label="Date"
            )
    pd.DataFrame(
        convergence_issues,
        columns=["model", "confidence_level", "date", "status", "message"],
    ).to_csv(output_dir / "convergence_diagnostics.csv", index=False)
    backtests = pd.concat(
        [summarise_all_backtests(r) for r in results.values()], ignore_index=True
    )
    backtests.to_csv(output_dir / "backtesting_results.csv", index=False)
    for level in config.confidence_levels:
        create_clean_backtesting_table(backtests, level).to_csv(
            output_dir / f"backtesting_{level:.0%}_formatted.csv", index=False
        )
    if plots:
        save_figures(portfolio, backtests, results, output_dir / "figures")
    manifest = {
        "garch_nonconverged_fits": len(convergence_issues),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ["numpy", "pandas", "scipy", "arch", "yfinance", "matplotlib"]
        },
        "tickers": config.tickers,
        "initial_value": config.initial_value,
        "window": window,
        "confidence_levels": config.confidence_levels,
        "models": model_names,
        "price_rows": len(prices),
        "portfolio_rows": len(portfolio),
        "first_price_date": str(prices.index.min()),
        "last_price_date": str(prices.index.max()),
        "prices_sha256": hashlib.sha256(
            (output_dir / "prices.csv").read_bytes()
        ).hexdigest(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Completed. Results: {output_dir.resolve()}", flush=True)
    return backtests


def save_figures(portfolio, backtests, results, figure_dir):
    from . import plotting

    figure_dir.mkdir()
    plotting.plot_portfolio_value(portfolio, figure_dir)
    plotting.plot_portfolio_returns(portfolio, figure_dir)
    plotting.plot_exceedance_rate_95(backtests, figure_dir)
    plotting.plot_exceedance_rate_99(backtests, figure_dir)
    for level in config.confidence_levels:
        for name in ["historical", "garch_t"]:
            if (name, level) in results:
                result = results[name, level]
                plotting.plot_var_model(
                    result,
                    f"{result['model'].iloc[0]} at {level:.0%} Confidence Level",
                    f"{name}_var_{int(level*100)}.png",
                    figure_dir,
                )
        if ("historical", level) in results and ("garch_t", level) in results:
            function = (
                plotting.plot_rolling_95 if level == 0.95 else plotting.plot_rolling_99
            )
            function(
                results["historical", level], results["garch_t", level], figure_dir
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce the global equity ETF VaR study."
    )
    parser.add_argument(
        "--prices",
        type=Path,
        help="Offline adjusted-price CSV: Date and all ten ticker columns.",
    )
    parser.add_argument(
        "--start", default=config.start_date, help="Inclusive price start date."
    )
    parser.add_argument(
        "--end", default=config.end_date, help="Exclusive price end date."
    )
    parser.add_argument("--window", type=int, default=config.window)
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), default=list(MODELS)
    )
    parser.add_argument(
        "--output", type=Path, help="New output directory (must not exist)."
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.window < 2:
        parser.error("--window must be at least 2")
    try:
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
        if start >= end:
            parser.error("--start must be earlier than --end")
        if args.prices:
            prices = pd.read_csv(args.prices, index_col="Date", parse_dates=["Date"])
        else:
            print("Downloading adjusted ETF prices from Yahoo Finance...", flush=True)
            prices = load_prices(config.tickers, args.start, args.end)
        prices = prices.loc[(prices.index >= start) & (prices.index < end)]
        output = args.output or Path("outputs") / datetime.now(UTC).strftime(
            "run-%Y%m%d-%H%M%S-%f"
        )
        run_analysis(prices, output, args.window, args.models, not args.no_plots)
    except (ValueError, FileExistsError, KeyError) as exc:
        parser.exit(1, f"Analysis could not start: {exc}\n")
