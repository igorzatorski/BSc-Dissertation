"""Shared forecast schema and strict breach convention."""

import pandas as pd


def make_var_output(portfolio, var_threshold, model_name, confidence_level, window):
    """Align forecasts with returns and label strict lower-tail exceedances."""
    result = pd.DataFrame(index=portfolio.index)

    result["price"] = portfolio["price"]
    result["log_ret"] = portfolio["log_ret"]
    result["var_threshold"] = var_threshold

    result = result.dropna(subset=["var_threshold"]).copy()

    result["var_positive"] = -result["var_threshold"]
    result["exceedance"] = result["log_ret"] < result["var_threshold"]

    result["model"] = model_name
    result["confidence_level"] = confidence_level
    result["tail_prob"] = 1 - confidence_level
    result["window"] = window

    return result
