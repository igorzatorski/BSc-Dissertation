"""Original dissertation figures, saved without blocking GUI windows."""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_var_model(var_result, title, filename, figure_dir, y_limits=None):
    data = var_result.dropna(subset=["log_ret", "var_threshold", "exceedance"]).copy()
    data["exceedance"] = data["exceedance"].astype(bool)

    exceedances = data[data["exceedance"]]

    _, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        data.index, data["log_ret"], linewidth=0.7, alpha=0.7, label="Daily log returns"
    )

    ax.plot(data.index, data["var_threshold"], linewidth=1.2, label="VaR threshold")

    ax.scatter(
        exceedances.index, exceedances["log_ret"], marker="x", s=28, label="Exceedances"
    )

    ax.axhline(0, linewidth=0.8, alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily log return (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    if y_limits is not None:
        ax.set_ylim(y_limits)

    ax.margins(x=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{figure_dir}/{filename}", dpi=300, bbox_inches="tight")
    plt.close()


def plot_portfolio_value(portfolio, figure_dir):
    # 30: Plot portfolio value over time

    _, ax = plt.subplots(figsize=(12, 5))

    ax.plot(portfolio.index, portfolio["price"] / 1_000_000, linewidth=1.2)

    ax.set_title("Global Equity ETF Portfolio Value Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD millions)")

    ax.grid(True, alpha=0.3)
    ax.margins(x=0)

    plt.tight_layout()
    plt.savefig(
        f"{figure_dir}/portfolio_value_over_time.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_portfolio_returns(portfolio, figure_dir):
    # 31: Plot daily portfolio log returns

    _, ax = plt.subplots(figsize=(12, 5))

    ax.plot(portfolio.index, portfolio["log_ret"], linewidth=0.7)

    ax.axhline(0, linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_title("Global Equity ETF Portfolio Daily Log Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Log Return (%)")

    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.grid(True, alpha=0.3)
    ax.margins(x=0)

    plt.tight_layout()
    plt.savefig(f"{figure_dir}/portfolio_log_returns.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_exceedance_rate_95(backtesting_results, figure_dir):
    # 37: Observed vs expected exceedance rates for 95% VaR

    bt_95 = backtesting_results[backtesting_results["confidence_level"] == 0.95].copy()

    bt_95 = bt_95.sort_values("coverage_error").reset_index(drop=True)

    plt.figure(figsize=(12, 6))

    plt.bar(bt_95["model"], bt_95["exceedance_rate"], label="Observed exceedance rate")

    plt.axhline(0.05, linestyle="--", linewidth=1.5, label="Expected exceedance rate")

    plt.title("Observed vs Expected Exceedance Rate for 95% VaR")
    plt.xlabel("Model")
    plt.ylabel("Exceedance Rate")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{figure_dir}/exceedance_rate_95.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_exceedance_rate_99(backtesting_results, figure_dir):
    # 38: Observed vs expected exceedance rates for 99% VaR

    bt_99 = backtesting_results[backtesting_results["confidence_level"] == 0.99].copy()

    bt_99 = bt_99.sort_values("coverage_error").reset_index(drop=True)

    plt.figure(figsize=(12, 6))

    plt.bar(bt_99["model"], bt_99["exceedance_rate"], label="Observed exceedance rate")

    plt.axhline(0.01, linestyle="--", linewidth=1.5, label="Expected exceedance rate")

    plt.title("Observed vs Expected Exceedance Rate for 99% VaR")
    plt.xlabel("Model")
    plt.ylabel("Exceedance Rate")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{figure_dir}/exceedance_rate_99.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_rolling_95(hist_95, garch_t_95, figure_dir):
    # 39: Rolling 252-day exceedances for Historical VaR and GARCH-t VaR

    if min(len(hist_95), len(garch_t_95)) < 253:
        warnings.warn(
            "Skipping 95% rolling exceedance chart: need at least 253 forecasts "
            "for two complete 252-day windows.",
            UserWarning,
            stacklevel=2,
        )
        return

    rolling_window = 252
    expected_exceedances_95 = rolling_window * 0.05

    hist_95_rolling_exceedances = (
        hist_95["exceedance"].astype(int).rolling(rolling_window).sum()
    )

    garch_t_95_rolling_exceedances = (
        garch_t_95["exceedance"].astype(int).rolling(rolling_window).sum()
    )

    _, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        hist_95.index,
        hist_95_rolling_exceedances,
        linewidth=1.2,
        label="Historical VaR",
    )

    ax.plot(
        garch_t_95.index,
        garch_t_95_rolling_exceedances,
        linewidth=1.2,
        label="GARCH(1,1)-Student-t VaR",
    )

    ax.axhline(
        expected_exceedances_95,
        linestyle="--",
        linewidth=1.3,
        label=f"Expected exceedances = {expected_exceedances_95:.1f}",
    )

    ax.set_title("Rolling 252-Day Number of Exceedances for 95% VaR")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Exceedances")

    ax.set_ylim(bottom=0)
    ax.margins(x=0)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{figure_dir}/rolling_exceedances_95.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_rolling_99(hist_99, garch_t_99, figure_dir):
    # 40: Rolling 252-day exceedances for Historical VaR and GARCH-t VaR at 99%

    if min(len(hist_99), len(garch_t_99)) < 253:
        warnings.warn(
            "Skipping 99% rolling exceedance chart: need at least 253 forecasts "
            "for two complete 252-day windows.",
            UserWarning,
            stacklevel=2,
        )
        return

    rolling_window = 252
    expected_exceedances_99 = rolling_window * 0.01

    hist_99_rolling_exceedances = (
        hist_99["exceedance"].astype(int).rolling(rolling_window).sum()
    )

    garch_t_99_rolling_exceedances = (
        garch_t_99["exceedance"].astype(int).rolling(rolling_window).sum()
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        hist_99.index,
        hist_99_rolling_exceedances,
        linewidth=1.2,
        label="Historical VaR",
    )

    plt.plot(
        garch_t_99.index,
        garch_t_99_rolling_exceedances,
        linewidth=1.2,
        label="GARCH(1,1)-Student-t VaR",
    )

    plt.axhline(
        expected_exceedances_99,
        linestyle="--",
        linewidth=1.5,
        label=f"Expected exceedances {expected_exceedances_99:.1f}",
    )

    plt.title("Rolling 252-Day Number of Exceedances for 99% VaR")
    plt.xlabel("Date")
    plt.ylabel("Number of Exceedances")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"{figure_dir}/rolling_exceedances_99.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
