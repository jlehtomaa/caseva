import numpy as np
import casadi as ca
from caseva.utils import is_almost_zero


class GenPareto:

    num_params = 2

    @staticmethod
    def cdf(x: np.ndarray, theta) -> np.ndarray:
        """Cumulative distribution function for Gen-Pareto distribution.

        Parameters
        ----------
        x : np.ndarray
            Sample quantiles to evaluate.

        Returns
        -------
        np.ndarray
            Cumulative distribution function value between 0 and 1.

        Raises
        ------
        ValueError
            If any input value is outside of the GPD domain when
            the shape parameter is negative.

        Notes
        -----
        See Coles (2001) p.75 eq. (4.2) and p.76 eq. (4.4). The `x` values
        correspond to the threshold excess values.
        """

        scale, shape = theta

        if (x <= 0).any():
            raise ValueError("Exceedances must be strictly greater than zero.")

        if shape < 0:
            support_upper_limit = -scale / shape
            if np.any(x > support_upper_limit):
                raise ValueError("Input value outside of support.")
            return 1. - np.exp(-x / scale)

        return 1. - (1. + shape * x / scale) ** (-1. / shape)

    @staticmethod
    def pdf(x: np.ndarray, theta) -> np.ndarray:
        """Probability density function for Gen-Pareto distribution.

        Parameters
        ----------
        x : np.ndarray
            Sample points.

        Returns
        -------
        np.ndarray
            Probability density values.

        Notes
        -----
        Hoskins p. 339 eq. (2).
        """

        scale, shape = theta

        if is_almost_zero(shape):
            return np.exp(-x / scale) / scale

        return ((1. + shape * x / scale) ** (-(1. / shape + 1.))) / scale

    @staticmethod
    def quantile(theta: ca.MX, proba: ca.MX) -> ca.MX:
        """Symbolic expression for the Gen-Pareto distribution quantiles.

        Parameters
        ----------
        theta : ca.MX
            Symbolic placeholder for the maximum likelihood parameters.
        proba : ca.MX
            Non-exceedance probability.

        Notes
        -----
        Coles 2001 p.84
        Hosking & Wallis (1987)
        """

        scale, shape = ca.vertsplit(theta)

        shape_zero = - scale * ca.log(1 - proba)
        shape_nonzero = - (scale / shape) * (1. - (1. - proba) ** (-shape))

        return ca.if_else(is_almost_zero(shape), shape_zero, shape_nonzero)
