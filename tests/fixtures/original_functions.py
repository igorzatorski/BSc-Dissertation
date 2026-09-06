"""Frozen numerical baseline extracted before notebook removal. Test use only."""

# 1: Importing all required libraries
import numpy as np
import pandas as pd
from arch.univariate import arch_model
from scipy.stats import chi2, norm, t


def build_equal_weighted_portfolio(prices, initial_value):
    # Calculate daily simple returns for each ETF
    individual_returns = prices.pct_change().dropna()

    # Create equal weights for all ETFs
    n_assets = individual_returns.shape[1]
    weights = pd.Series(1 / n_assets, index=individual_returns.columns)

    # Daily rebalanced portfolio return
    portfolio_simple_return = individual_returns.dot(weights)

    # Convert daily portfolio returns into portfolio value
    portfolio_value = initial_value * (1 + portfolio_simple_return).cumprod()

    # Store portfolio value and returns in one DataFrame
    portfolio = pd.DataFrame(
        {"price": portfolio_value, "simple_ret": portfolio_simple_return}
    )

    # Calculate log returns for VaR modelling
    portfolio["log_ret"] = np.log(portfolio["price"] / portfolio["price"].shift(1))

    # Remove the first missing log return
    portfolio = portfolio.dropna()

    return portfolio


def make_var_output(portfolio, var_threshold, model_name, confidence_level, window):
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


def historical_var(portfolio, window, confidence_level):
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


def normal_var(portfolio, window, confidence_level):
    # Convert confidence level into tail probability
    tail_prob = 1 - confidence_level

    # Calculate rolling mean using past returns only
    rolling_mean = portfolio["log_ret"].shift(1).rolling(window=window).mean()

    # Calculate rolling standard deviation using past returns only
    rolling_std = portfolio["log_ret"].shift(1).rolling(window=window).std()

    # Normal left-tail quantile
    z_score = norm.ppf(tail_prob)

    # Parametric Normal VaR threshold
    var_threshold = rolling_mean + rolling_std * z_score

    # Return results in the standard VaR output format
    return make_var_output(
        portfolio=portfolio,
        var_threshold=var_threshold,
        model_name="Normal VaR",
        confidence_level=confidence_level,
        window=window,
    )


def student_t_var(portfolio, window, confidence_level):
    # Convert confidence level into tail probability
    tail_prob = 1 - confidence_level

    # Prepare empty Series for VaR threshold
    var_threshold = pd.Series(index=portfolio.index, dtype=float)

    # Rolling estimation of Student-t parameters
    for i in range(window, len(portfolio)):
        # Use only past returns to avoid look-ahead bias
        sample = portfolio["log_ret"].iloc[i - window : i]

        # Fit Student-t distribution to the rolling window
        nu, loc, scale = t.fit(sample)

        # Calculate left-tail quantile of fitted Student-t distribution
        var_threshold.iloc[i] = t.ppf(tail_prob, df=nu, loc=loc, scale=scale)

    # Return results in the standard VaR output format
    return make_var_output(
        portfolio=portfolio,
        var_threshold=var_threshold,
        model_name="Student-t VaR",
        confidence_level=confidence_level,
        window=window,
    )


def garch_normal_var(portfolio, window, confidence_level):
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


def summarise_var_results(var_result):
    model = var_result["model"].iloc[0]
    confidence_level = var_result["confidence_level"].iloc[0]
    tail_prob = var_result["tail_prob"].iloc[0]
    window = var_result["window"].iloc[0]

    n_obs = len(var_result)
    n_exceed = int(var_result["exceedance"].sum())
    expected_exceed = n_obs * tail_prob
    exceedance_rate = n_exceed / n_obs
    expected_rate = tail_prob
    empirical_coverage = 1 - exceedance_rate
    average_var = var_result["var_positive"].mean()

    summary = pd.DataFrame(
        {
            "model": [model],
            "confidence_level": [confidence_level],
            "window": [window],
            "n_obs": [n_obs],
            "n_exceed": [n_exceed],
            "expected_exceed": [expected_exceed],
            "exceedance_rate": [exceedance_rate],
            "expected_rate": [expected_rate],
            "empirical_coverage": [empirical_coverage],
            "average_var": [average_var],
        }
    )

    return summary


def kupiec_test(var_result):
    # Extract exceedance indicators as 0/1 values
    exceedances = var_result["exceedance"].astype(int)

    # Number of observations and number of exceedances
    n_obs = len(exceedances)
    n_exceed = int(exceedances.sum())

    # Expected exceedance probability
    tail_prob = var_result["tail_prob"].iloc[0]

    # Observed exceedance probability
    observed_prob = n_exceed / n_obs

    # Helper function for binomial log-likelihood
    def binomial_log_likelihood(prob, n_exceed, n_obs):
        if prob == 0:
            return 0 if n_exceed == 0 else -np.inf
        if prob == 1:
            return 0 if n_exceed == n_obs else -np.inf

        return n_exceed * np.log(prob) + (n_obs - n_exceed) * np.log(1 - prob)

    # Log-likelihood under the null hypothesis: exceedance probability = tail_prob
    log_likelihood_null = binomial_log_likelihood(tail_prob, n_exceed, n_obs)

    # Log-likelihood under the alternative hypothesis: exceedance probability = observed_prob
    log_likelihood_alt = binomial_log_likelihood(observed_prob, n_exceed, n_obs)

    # Kupiec likelihood ratio statistic
    lr_uc = -2 * (log_likelihood_null - log_likelihood_alt)

    # p-value from chi-square distribution with 1 degree of freedom
    p_value = 1 - chi2.cdf(lr_uc, df=1)

    return pd.DataFrame(
        {
            "kupiec_lr": [lr_uc],
            "kupiec_p_value": [p_value],
            "kupiec_reject_5pct": [p_value < 0.05],
        }
    )


def summarise_var_results_with_kupiec(var_result):
    summary = summarise_var_results(var_result)
    kupiec = kupiec_test(var_result)

    return pd.concat(
        [summary.reset_index(drop=True), kupiec.reset_index(drop=True)], axis=1
    )


def christoffersen_test(var_result):
    # Convert exceedance indicator to integer 0/1
    exceedances = var_result["exceedance"].astype(int).values

    # Previous and current exceedance indicators
    previous = exceedances[:-1]
    current = exceedances[1:]

    # Transition counts
    n00 = np.sum((previous == 0) & (current == 0))
    n01 = np.sum((previous == 0) & (current == 1))
    n10 = np.sum((previous == 1) & (current == 0))
    n11 = np.sum((previous == 1) & (current == 1))

    # Safe log-likelihood helper
    def safe_log_likelihood(count_success, count_failure, prob):
        if prob <= 0:
            return 0 if count_success == 0 else -np.inf
        if prob >= 1:
            return 0 if count_failure == 0 else -np.inf

        return count_success * np.log(prob) + count_failure * np.log(1 - prob)

    # Unconditional exceedance probability in transition sample
    total_transitions = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total_transitions

    # Conditional exceedance probabilities
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0

    # Log-likelihood under independence
    log_likelihood_independent = safe_log_likelihood(n01 + n11, n00 + n10, pi)

    # Log-likelihood under Markov dependence
    log_likelihood_dependent = safe_log_likelihood(n01, n00, pi0) + safe_log_likelihood(
        n11, n10, pi1
    )

    # Christoffersen independence likelihood ratio statistic
    lr_ind = -2 * (log_likelihood_independent - log_likelihood_dependent)
    p_value_ind = 1 - chi2.cdf(lr_ind, df=1)

    # Kupiec unconditional coverage statistic
    kupiec = kupiec_test(var_result)
    lr_uc = kupiec["kupiec_lr"].iloc[0]

    # Conditional coverage statistic = unconditional coverage + independence
    lr_cc = lr_uc + lr_ind
    p_value_cc = 1 - chi2.cdf(lr_cc, df=2)

    return pd.DataFrame(
        {
            "n00": [n00],
            "n01": [n01],
            "n10": [n10],
            "n11": [n11],
            "pi0": [pi0],
            "pi1": [pi1],
            "christoffersen_ind_lr": [lr_ind],
            "christoffersen_ind_p_value": [p_value_ind],
            "christoffersen_ind_reject_5pct": [p_value_ind < 0.05],
            "christoffersen_cc_lr": [lr_cc],
            "christoffersen_cc_p_value": [p_value_cc],
            "christoffersen_cc_reject_5pct": [p_value_cc < 0.05],
        }
    )


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


def summarise_all_backtests(var_result):
    summary = summarise_var_results(var_result)
    kupiec = kupiec_test(var_result)
    christoffersen = christoffersen_test(var_result)

    result = pd.concat(
        [
            summary.reset_index(drop=True),
            kupiec.reset_index(drop=True),
            christoffersen.reset_index(drop=True),
        ],
        axis=1,
    )

    result["coverage_error"] = (
        result["exceedance_rate"] - result["expected_rate"]
    ).abs()

    return result
