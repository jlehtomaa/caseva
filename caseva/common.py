"""
Common helper functionalities used by (almost) all of the implemented models.
"""

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


def build_return_level_func(
    num_mle_params: int,
    return_level_expr: ca.MX
) -> ca.Function:
    """Build function evaluating return level and uncertainty with delta method.

    Parameters
    ----------
    num_mle_params : int
        Number of maximum likelihood estimated parameters.
    return_level_expr : ca.MX
        Symbolic return level expression.

    Returns
    -------
    ca.Function
        A function evaluating return level and its confidence interval as a
        function of MLE params, exceedance probability, and covariance.

    Notes
    -----
    See Coles (2001) p.33, Chapter 2.6: Parametric Modeling.
    """

    proba = ca.MX.sym("proba", 1)
    theta = ca.MX.sym("theta", num_mle_params)
    covar = ca.MX.sym("covar", num_mle_params, num_mle_params)

    level = return_level_expr(theta, proba)
    grad = ca.jacobian(level, theta)
    error = 1.96 * ca.sqrt(grad @ covar @ grad.T)

    return ca.Function(
        "return_level_fn",
        [theta, proba, covar],
        [level, level+error, level-error],
        ["theta", "proba", "covar"],
        ["level", "upper", "lower"]
        )


def empirical_return_periods(
    arr,
    num_years: int
) -> np.ndarray:
    """Calculate the empirical return period for each element in an array.

    Parameters
    ----------
    arr : np.ndarray | list
        Input data values.
    num_years : int
        Number of years of the observation window.

    Returns
    -------
    numpy.ndarray
        Empirical return period for each element in the original array.
    """

    arr = np.array(arr)
    assert arr.ndim == 1, "Input array must have exactly one dimension."

    num_exceedances = np.sum(arr[:, None] <= arr, axis=1)
    annual_exceedance_freq = num_exceedances / num_years
    return 1. / annual_exceedance_freq


def ca2np(arr: ca.DM) -> np.ndarray:
    """Casadi data array as a 1-dimensional numpy array."""
    return np.array(arr).ravel()


def mean_residual_life(data, thresholds=None, conf_level=None, ax=None):

    if thresholds is None:

        smallest = data.min()
        second_largest = data.sort_values().iloc[-2]
        thresholds = np.linspace(smallest, second_largest, 100, endpoint=True)

    if conf_level is None:
        conf_level = 0.95

    avg_exceedances, conf_intervals = [], []
    # Use approx normality of sample means for confidence intervals.
    for threshold in thresholds:

        exceedances = data[data > threshold] - threshold

        avg_exceedances.append(
            exceedances.mean()
        )

        conf_intervals.append(
            scipy.stats.norm.interval(
                alpha=conf_level,
                loc=exceedances.mean(),
                scale=exceedances.std() / np.sqrt(len(exceedances))
            )
        )

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(thresholds, avg_exceedances)
    ax.fill_between(thresholds, *np.transpose(conf_intervals), alpha=0.50)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean excess")

    return ax