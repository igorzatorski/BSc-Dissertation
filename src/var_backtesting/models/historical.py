"""Historical one-day VaR forecasts using past returns only."""

from .base import make_var_output


def historical_var(portfolio, window, confidence_level):
    """Forecast empirical lower-tail quantiles from the preceding window."""
    # Convert confidence level into tail probability
    tail_prob = 1 - confidence_level

    # Calculate historical VaR threshold using past rolling returns only
    var_threshold = (
        portfolio["log_ret"].shift(1).rolling(window=window).quantile(tail_prob)
    )

    # Return results in the standard VaR output format
    return make_var_output(
        portfolio=portfolio,
        var_threshold=var_threshold,
        model_name="Historical VaR",
        confidence_level=confidence_level,
        window=window,
    )
