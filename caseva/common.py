"""
Common helper functionalities used by (almost) all of the implemented models.
"""

import numpy as np
import pandas as pd
import casadi as ca


def build_return_level_func(
    num_mle_params: int,
    return_level_expr: ca.MX
) -> ca.Function:
    """Build function for return level and uncertainty with delta method.

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


def empirical_return_periods(values, num_years):

    sorted_values = np.sort(values)[::-1]  # descending
    ranking = np.arange(1, len(sorted_values) + 1)

    return_periods = (num_years + 1) / ranking
    return pd.Series(data=sorted_values, index=return_periods)


def ca2np(arr: ca.DM) -> np.ndarray:
    """Casadi data array as a 1-dimensional numpy array."""
    return np.array(arr).ravel()
