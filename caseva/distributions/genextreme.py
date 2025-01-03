"""Generalized Extreme Value Distribution functions: cdf, pdf, quantile."""

import numpy as np
import casadi as ca
from caseva.utils import is_almost_zero


class GenExtreme:

    num_params = 3  # location, scale, shape

    @staticmethod
    def cdf(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Cumulative distribution function for GEV distribution.

        Parameters
        ----------
        x : np.ndarray
            Sample quantiles to evaluate.
        theta : np.ndarray
            Parameters of the fitted GEV distribution.

        Returns
        -------
        np.ndarray
            Cumulative distribution function value between 0 and 1.

        Notes
        -----
        https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution or
        Coles (2001) p. 47-48, (3.2).
        """

        loc, scale, shape = theta
        xnorm = (x - loc) / scale

        if is_almost_zero(shape):
            tx = np.exp(-xnorm)
        else:
            tx = (1. + shape * xnorm) ** (-1. / shape)

        return np.exp(-tx)

    @staticmethod
    def pdf(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Probability density function for GEV distribution.

        Parameters
        ----------
        x : np.ndarray
            Sample points.
        theta : np.ndarray
            Parameters of the fitted distribution.

        Returns
        -------
        np.ndarray
            Probability density values.

        Notes
        -----
        https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
        """
        loc, scale, shape = theta
        ynorm = (x - loc) / scale

        if is_almost_zero(shape):
            tx = np.exp(-ynorm)
        else:
            tx = (1. + shape * ynorm) ** (-1. / shape)

        return (1. / scale) * tx ** (shape + 1) * np.exp(-tx)

    @staticmethod
    def quantile(theta: ca.MX, proba: ca.MX) -> ca.MX:
        """Symbolic expression for the GEV distribution quantiles.

        Parameters
        ----------
        theta : ca.MX
            Symbolic placeholder for the maximum likelihood parameters.
        proba : ca.MX
            Non-exceedance probability.

        Returns
        -------
        ca.MX
            Casadi symbolic quantile expression. Represents a value such that
            a random variable has a `proba` probability of being less than or
            equal to that value.

        Notes
        -----
        For details, see
        - Coles (2001) p. 49 eq. (3.4)
        - Coles (2001) p. 56 eq. (3.10)

        Note that the above equations are written in terms of the *extreme*
        quantiles, using 1 - proba, such that `proba` refers to an exceedance
        probability. Here, instead, the usual quantile definition is used.
        """

        log_prob = -ca.log(proba)
        loc, scale, shape = ca.vertsplit(theta)

        shape_zero = loc - scale * ca.log(log_prob)
        shape_nonz = loc - (scale / shape) * (1. - log_prob ** (-shape))

        return ca.if_else(is_almost_zero(shape), shape_zero, shape_nonz)
