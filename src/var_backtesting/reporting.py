"""Display formatting; numeric results stay separate from presentation."""


def format_comparison_table(df):
    formatted = df.copy()

    formatted["expected_exceed"] = formatted["expected_exceed"].map(
        lambda x: f"{x:.2f}"
    )
    formatted["exceedance_rate"] = formatted["exceedance_rate"].map(
        lambda x: f"{x:.2%}"
    )
    formatted["expected_rate"] = formatted["expected_rate"].map(lambda x: f"{x:.2%}")
    formatted["coverage_error"] = formatted["coverage_error"].map(lambda x: f"{x:.2%}")
    formatted["empirical_coverage"] = formatted["empirical_coverage"].map(
        lambda x: f"{x:.2%}"
    )
    formatted["average_var"] = formatted["average_var"].map(lambda x: f"{x:.2%}")

    return formatted


def format_kupiec_table(df):
    formatted = df.copy()

    formatted["expected_exceed"] = formatted["expected_exceed"].map(
        lambda x: f"{x:.2f}"
    )
    formatted["exceedance_rate"] = formatted["exceedance_rate"].map(
        lambda x: f"{x:.2%}"
    )
    formatted["expected_rate"] = formatted["expected_rate"].map(lambda x: f"{x:.2%}")
    formatted["coverage_error"] = formatted["coverage_error"].map(lambda x: f"{x:.2%}")
    formatted["empirical_coverage"] = formatted["empirical_coverage"].map(
        lambda x: f"{x:.2%}"
    )
    formatted["average_var"] = formatted["average_var"].map(lambda x: f"{x:.2%}")
    formatted["kupiec_lr"] = formatted["kupiec_lr"].map(lambda x: f"{x:.4f}")
    formatted["kupiec_p_value"] = formatted["kupiec_p_value"].map(lambda x: f"{x:.4g}")

    return formatted


def format_p_value(p):
    if p < 0.001:
        return "<0.001"
    else:
        return f"{p:.4f}"


def format_rejection(reject):
    if reject:
        return "Reject"
    else:
        return "Do not reject"


def create_clean_backtesting_table(backtesting_results, confidence_level):
    """Select and format one confidence level for human-readable reporting."""
    table = backtesting_results[
        backtesting_results["confidence_level"] == confidence_level
    ].copy()

    table = table.sort_values("coverage_error").reset_index(drop=True)

    columns_to_keep = [
        "model",
        "n_exceed",
        "expected_exceed",
        "exceedance_rate",
        "coverage_error",
        "average_var",
        "kupiec_p_value",
        "kupiec_reject_5pct",
        "christoffersen_ind_p_value",
        "christoffersen_ind_reject_5pct",
        "christoffersen_cc_p_value",
        "christoffersen_cc_reject_5pct",
    ]

    table = table[columns_to_keep]

    table = table.rename(
        columns={
            "model": "Model",
            "n_exceed": "Exceedances",
            "expected_exceed": "Expected exceedances",
            "exceedance_rate": "Exceedance rate",
            "coverage_error": "Coverage error",
            "average_var": "Average VaR",
            "kupiec_p_value": "Kupiec p-value",
            "kupiec_reject_5pct": "Kupiec decision",
            "christoffersen_ind_p_value": "Independence p-value",
            "christoffersen_ind_reject_5pct": "Independence decision",
            "christoffersen_cc_p_value": "Conditional coverage p-value",
            "christoffersen_cc_reject_5pct": "Conditional coverage decision",
        }
    )

    table["Expected exceedances"] = table["Expected exceedances"].map(
        lambda x: f"{x:.2f}"
    )
    table["Exceedance rate"] = table["Exceedance rate"].map(lambda x: f"{x:.2%}")
    table["Coverage error"] = table["Coverage error"].map(lambda x: f"{x:.2%}")
    table["Average VaR"] = table["Average VaR"].map(lambda x: f"{x:.2%}")

    table["Kupiec p-value"] = table["Kupiec p-value"].map(format_p_value)
    table["Independence p-value"] = table["Independence p-value"].map(format_p_value)
    table["Conditional coverage p-value"] = table["Conditional coverage p-value"].map(
        format_p_value
    )

    table["Kupiec decision"] = table["Kupiec decision"].map(format_rejection)
    table["Independence decision"] = table["Independence decision"].map(
        format_rejection
    )
    table["Conditional coverage decision"] = table["Conditional coverage decision"].map(
        format_rejection
    )

    return table
