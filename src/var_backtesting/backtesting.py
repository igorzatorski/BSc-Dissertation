"""Coverage summaries and Kupiec/Christoffersen likelihood-ratio tests."""

import numpy as np
import pandas as pd
from scipy.stats import chi2


def summarise_var_results(var_result):
    """Summarise forecast coverage and the mean negated VaR threshold."""
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
    """Test unconditional coverage against the model tail probability."""
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
    """Combine descriptive coverage statistics with the Kupiec test."""
    summary = summarise_var_results(var_result)
    kupiec = kupiec_test(var_result)

    return pd.concat(
        [summary.reset_index(drop=True), kupiec.reset_index(drop=True)], axis=1
    )


def christoffersen_test(var_result):
    """Test breach independence and joint conditional coverage."""
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


def summarise_all_backtests(var_result):
    """Combine coverage, Kupiec and Christoffersen statistics in one row."""
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
