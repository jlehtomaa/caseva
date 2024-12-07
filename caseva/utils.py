"""Helper functions used in multiple modules."""

import numpy as np
import casadi as ca


def ca2np(arr: ca.DM) -> np.ndarray:
    """Return casadi data matrix as a 1-dimensional numpy array."""
    return np.array(arr).ravel()


def is_almost_zero(val: float, tol: float = 1e-8) -> bool:
    return ca.fabs(val) < tol
