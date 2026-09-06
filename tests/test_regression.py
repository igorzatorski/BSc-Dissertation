"""Compare every model and test against the original notebook on identical data."""

import numpy as np
import pandas as pd
import pytest
from fixtures import original_functions as original

from var_backtesting import config
from var_backtesting.backtesting import (
    christoffersen_test,
    kupiec_test,
    summarise_all_backtests,
)
from var_backtesting.data import validate_prices
from var_backtesting.models import MODELS
from var_backtesting.pipeline import run_analysis
from var_backtesting.portfolio import build_equal_weighted_portfolio


@pytest.fixture
def prices():
    rng = np.random.default_rng(2026)
    values = 100 * np.exp(np.cumsum(rng.standard_t(6, size=(272, 10)) * 0.01, axis=0))
    return pd.DataFrame(
        values, index=pd.bdate_range("2020-01-01", periods=272), columns=config.tickers
    )


def test_portfolio_matches_notebook(prices):
    actual = build_equal_weighted_portfolio(prices, 1_000_000)
    pd.testing.assert_frame_equal(
        actual, original.build_equal_weighted_portfolio(prices, 1_000_000)
    )
    assert len(actual) == len(prices) - 2


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("level", [0.95, 0.99])
def test_models_and_backtests_match_notebook(prices, model, level):
    portfolio = build_equal_weighted_portfolio(prices, 1_000_000)
    function = MODELS[model]
    actual = function(portfolio, 252, level)
    expected = getattr(original, function.__name__)(portfolio, 252, level)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(
        summarise_all_backtests(actual),
        original.summarise_all_backtests(expected),
        check_exact=True,
    )
    assert len(actual) == 18
    # Changing the realised return on a forecast date cannot alter that day's forecast.
    changed = portfolio.copy()
    changed.loc[actual.index[-1], "log_ret"] = -0.9
    assert (
        function(changed, 252, level)["var_threshold"].iloc[-1]
        == actual["var_threshold"].iloc[-1]
    )


@pytest.mark.parametrize("breach", [False, True])
def test_degenerate_breach_sequences(breach):
    frame = pd.DataFrame({"exceedance": [breach] * 20, "tail_prob": [0.05] * 20})
    assert np.isfinite(kupiec_test(frame)["kupiec_lr"].iloc[0])
    assert christoffersen_test(frame)["christoffersen_ind_lr"].iloc[0] == 0


def test_reject_missing_universe(prices):
    with pytest.raises(ValueError, match="all configured"):
        validate_prices(prices.drop(columns="SPY"), config.tickers)


def test_pipeline_outputs_and_overwrite_protection(prices, tmp_path):
    out = tmp_path / "run"
    result = run_analysis(prices, out, models=["historical", "normal"], plots=False)
    assert len(result) == 4
    assert len(list((out / "forecasts").glob("*.csv"))) == 4
    assert (out / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        run_analysis(prices, out, models=["historical"], plots=False)
