"""
Common helper functionalities used by (almost) all of the implemented models.
"""

import numpy as np
import casadi as ca

def ca2np(arr: ca.DM) -> np.ndarray:
    """Casadi data array as a 1-dimensional numpy array."""
    return np.array(arr).ravel()
