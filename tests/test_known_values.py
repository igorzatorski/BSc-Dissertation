"""Independent checks with small examples whose answers are known in advance."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from var_backtesting.backtesting import kupiec_test
from var_backtesting.models import historical_var, normal_var


def _portfolio(returns):
    index = pd.bdate_range("2025-01-01", periods=len(returns))
    return pd.DataFrame(
        {"price": 100 * np.exp(np.cumsum(returns)), "log_ret": returns},
        index=index,
    )


def test_historical_var_uses_only_the_previous_window():
    portfolio = _portfolio(np.array([-0.03, -0.01, 0.01, 0.02, -0.50]))
    result = historical_var(portfolio, window=4, confidence_level=0.75)

    expected = np.quantile([-0.03, -0.01, 0.01, 0.02], 0.25)
    assert np.isclose(result["var_threshold"].iloc[0], expected)
    assert result["exceedance"].iloc[0]


def test_normal_var_matches_the_closed_form_forecast():
    past = np.array([-0.02, -0.01, 0.01, 0.02])
    portfolio = _portfolio(np.append(past, 0.0))
    result = normal_var(portfolio, window=4, confidence_level=0.95)

    expected = past.mean() + past.std(ddof=1) * norm.ppf(0.05)
    assert np.isclose(result["var_threshold"].iloc[0], expected)


def test_kupiec_statistic_matches_direct_likelihood_calculation():
    exceedances = np.array([True] * 2 + [False] * 98)
    frame = pd.DataFrame({"exceedance": exceedances, "tail_prob": 0.05})
    result = kupiec_test(frame)

    observed = 0.02
    log_null = 2 * np.log(0.05) + 98 * np.log(0.95)
    log_alt = 2 * np.log(observed) + 98 * np.log(1 - observed)
    expected_lr = -2 * (log_null - log_alt)
    assert np.isclose(result["kupiec_lr"].iloc[0], expected_lr)
