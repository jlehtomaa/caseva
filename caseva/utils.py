"""Helper functions used in multiple modules."""

import numpy as np
import casadi as ca


def ca2np(arr: ca.DM) -> np.ndarray:
    """Return casadi data matrix as a 1-dimensional numpy array."""
    return np.array(arr).ravel()


def is_almost_zero(val: float, tol: float = 1e-8) -> bool:
    """Return True if `val` is within `tol` from zero and False otherwise."""
    return ca.fabs(val) < tol


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical cumulative distribution function.

    Parameters
    ----------
    values : np.ndarray
        Observed data values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (quantiles, probabilities): Quantiles are the sorted `values`,
        and probabilities are the empirical cumulative probabilities.
    """

    quantiles = np.sort(values)  # ascending
    probabilities = np.arange(1, len(quantiles) + 1) / (len(quantiles) + 1)

    return quantiles, probabilities
