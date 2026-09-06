"""Publication edge cases: honest diagnostics and meaningful rolling charts."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from var_backtesting import config, plotting
from var_backtesting.models import garch
from var_backtesting.pipeline import run_analysis


@pytest.mark.parametrize(
    "function", [plotting.plot_rolling_95, plotting.plot_rolling_99]
)
@pytest.mark.parametrize("n", [17, 252, 253])
def test_rolling_chart_minimum_sample(function, n, tmp_path):
    frame = pd.DataFrame(
        {"exceedance": np.arange(n) % 20 == 0},
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    if n < 253:
        with pytest.warns(UserWarning, match="Skipping"):
            function(frame, frame, tmp_path)
        assert not list(tmp_path.glob("*.png"))
    else:
        function(frame, frame, tmp_path)
        assert len(list(tmp_path.glob("*.png"))) == 1


@pytest.mark.parametrize("model", ["garch_normal", "garch_t"])
def test_nonconvergence_is_reported_and_saved(model, tmp_path):
    prices = pd.DataFrame(
        100 + np.arange(8)[:, None] * np.ones((1, 10)),
        index=pd.bdate_range("2020-01-01", periods=8),
        columns=config.tickers,
    )
    forecast = SimpleNamespace(
        mean=pd.DataFrame([[0.0]]), variance=pd.DataFrame([[1.0]])
    )
    fit = SimpleNamespace(
        convergence_flag=9,
        optimization_result=SimpleNamespace(message="Iteration limit"),
        params={"nu": 6.0},
        forecast=lambda **kwargs: forecast,
    )
    with patch.object(garch, "arch_model") as factory:
        factory.return_value.fit.return_value = fit
        with pytest.warns(RuntimeWarning, match="did not converge"):
            result = run_analysis(
                prices, tmp_path / "run", window=4, models=[model], plots=False
            )
    issues = pd.read_csv(tmp_path / "run/convergence_diagnostics.csv")
    assert len(issues) == 4
    assert set(issues.status) == {9}
    assert set(issues.model) == {model}
    assert set(issues.date) == set(prices.index[-2:].astype(str) + " 00:00:00")
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["garch_nonconverged_fits"] == 4
    assert (result.n_obs == 2).all()
