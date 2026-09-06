"""Garch one-day VaR forecasts using past returns only."""

import numpy as np
import pandas as pd
from arch.univariate import arch_model
from scipy.stats import norm, t

from .base import make_var_output


def garch_normal_var(portfolio, window, confidence_level):
    """Refit GARCH(1,1) with Normal innovations and forecast one day ahead."""
    # Convert confidence level into tail probability
    tail_prob = 1 - confidence_level

    # Prepare empty Series for VaR threshold
    var_threshold = pd.Series(index=portfolio.index, dtype=float)

    # Rolling GARCH estimation
    for i in range(window, len(portfolio)):
        # Use only past returns to avoid look-ahead bias
        # Multiply by 100 because the arch package works better with percentage returns
        sample = portfolio["log_ret"].iloc[i - window : i] * 100

        # Define GARCH(1,1) model with Normal innovations
        model = arch_model(
            sample, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False
        )

        # Fit the model
        fitted_model = model.fit(disp="off", update_freq=0, show_warning=False)

        # Forecast one-day-ahead conditional mean and variance
        forecast = fitted_model.forecast(horizon=1, reindex=False)

        mu_forecast = forecast.mean.iloc[-1, 0]
        sigma_forecast = np.sqrt(forecast.variance.iloc[-1, 0])

        # Normal left-tail quantile
        q = norm.ppf(tail_prob)

        # GARCH-Normal VaR threshold, converted back from percentage to decimal returns
        var_threshold.iloc[i] = (mu_forecast + sigma_forecast * q) / 100

    # Return results in the standard VaR output format
    return make_var_output(
        portfolio=portfolio,
        var_threshold=var_threshold,
        model_name="GARCH(1,1)-Normal VaR",
        confidence_level=confidence_level,
        window=window,
    )


def garch_t_var(portfolio, window, confidence_level):
    """Refit GARCH(1,1) with unit-variance Student-t innovations."""
    # Convert confidence level into tail probability
    tail_prob = 1 - confidence_level

    # Prepare empty Series for VaR threshold
    var_threshold = pd.Series(index=portfolio.index, dtype=float)

    # Rolling GARCH estimation
    for i in range(window, len(portfolio)):
        # Use only past returns to avoid look-ahead bias
        # Multiply by 100 because arch works better with percentage returns
        sample = portfolio["log_ret"].iloc[i - window : i] * 100

        # Define GARCH(1,1) model with Student-t innovations
        model = arch_model(
            sample, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False
        )

        # Fit the model
        fitted_model = model.fit(disp="off", update_freq=0, show_warning=False)

        # Forecast one-day-ahead conditional mean and variance
        forecast = fitted_model.forecast(horizon=1, reindex=False)

        mu_forecast = forecast.mean.iloc[-1, 0]
        sigma_forecast = np.sqrt(forecast.variance.iloc[-1, 0])

        # Extract Student-t degrees of freedom
        nu = fitted_model.params["nu"]

        # Student-t left-tail quantile
        # The arch package uses standardised Student-t innovations,
        # so the scipy t quantile is rescaled to have variance equal to 1.
        q = t.ppf(tail_prob, df=nu) * np.sqrt((nu - 2) / nu)

        # GARCH-t VaR threshold, converted back from percentage to decimal returns
        var_threshold.iloc[i] = (mu_forecast + sigma_forecast * q) / 100

    # Return results in the standard VaR output format
    return make_var_output(
        portfolio=portfolio,
        var_threshold=var_threshold,
        model_name="GARCH(1,1)-Student-t VaR",
        confidence_level=confidence_level,
        window=window,
    )
