"""
Implementation of the classical extreme value model with (annual) block maxima.
See Coles (2001) Chapter 3.
"""

import numpy as np
import casadi as ca

from caseva.optimizer import MLEOptimizer
from caseva.models import BaseModel
from caseva.common import build_return_level_func


OPTIM_BOUNDS = np.array([
    [-100, 100],  # Location, \mu
    [1e-8, 100],  # Scale, \sigma
    [-1, 100]])   # Shape, \xi
"""
See discussion in Coles (2001) p. 55 for the shape parameter constraints.
"""


class BlockMaximaModel(MLEOptimizer, BaseModel):
    """Classical extreme value model with annual block maxima."""

    num_params = 3

    def __init__(
        self,
        extremes: np.ndarray,
        max_optim_restarts: int = 0,
        seed: int = 0
    ):
        """

        Parameters
        ----------
        extremes : np.ndarray
            A 1d array of observed extreme values (annual maxima).
        max_optim_restarts : int, default = 0
            How many randomly initialized optimizer restarts to perform if no
            solution is found.
        seed : int, default = 0
            Seed for generating random optimizer restarts.
        """

        super().__init__(
            extremes=extremes,
            seed=seed,
            max_optim_restarts=max_optim_restarts,
            num_years=len(extremes),
            optim_bounds=OPTIM_BOUNDS
        )

        self.return_level_fn = build_return_level_func(
            self.num_params, self.return_level_expr)

    def constraints_fn(self, theta, extremes):
        """Builds the log likelihood constraints for the numerical optimizer.

        Parameters
        ----------
        theta : ca.MX
            Symbolic placeholder for the maximum likelihood parameter estimate.
        extremes : ca.DM
            Observed extreme values.

        Returns
        -------
        constr : list
            Collection of symbolic constraint expressions.

        Notes
        -----
        Coles (2001) p. 55 eq. (3.8). The parameter constraint must hold for
        all elements in `extremes`.
        """

        loc, scale, shape = ca.vertsplit(theta)

        constr = [
            (self.optim_bounds[:, 0] <= theta) <= self.optim_bounds[:, 1],
            1. + shape * ((extremes - loc) / scale) > 0.
        ]

        return constr

    @staticmethod
    def optimizer_initial_guess(extremes):
        """Derive the initial guess for the MLE optimization.

        Use the same value as in the 'ismev' R package that accompanies the
        Coles 2001 book: https://github.com/cran/ismev/blob/master/R/gev.R

        The scale_init is based on the method of moments for a Gumbel
        distribution.

        Parameters
        ----------
        extremes : array-like
            The extreme observations used for maximum likelihood estimation.
        """

        scale_init = np.sqrt(6. * np.var(extremes)) / np.pi
        loc_init = np.mean(extremes) - 0.57722 * scale_init
        shape_init = 0.1

        return [loc_init, scale_init, shape_init]

    def cdf(self, x):
        """GEV cumulative distribution function.

        Parameters
        ----------
        x : Union[float, np.ndarray]
            Sample point.

        Notes
        -----
        https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
        or
        Coles (2001) p. 47-48, (3.2).
        """

        loc, scale, shape = self.theta
        xnorm = (x - loc) / scale

        if np.abs(shape) < self.tiny:
            tx = np.exp(-xnorm)
        else:
            tx = (1. + shape * xnorm) ** (-1. / shape)

        return np.exp(-tx)

    def pdf(self, x):
        """GEV probability density function.

        Parameters
        ----------
        x : Union[float, np.ndarray]
            Sample point.

        Notes
        -----
        https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
        """
        loc, scale, shape = self.theta
        ynorm = (x - loc) / scale

        if np.abs(shape) < self.tiny:
            tx = np.exp(-ynorm)
        else:
            tx = (1. + shape * ynorm) ** (-1. / shape)

        return (1. / scale) * tx ** (shape + 1) * np.exp(-tx)

    def log_likelihood(self, theta, extremes):
        """GEV log likelihood function symbolic expression.

        Parameters
        ----------
        theta : ca.MX
            Maximum likelihood parameter estimate symbolic placeholder.
        extremes : ca.DM
            Extreme value obserations.

        Returns
        -------
        ca.MX
            Symbolic log likelihood expression.

        Notes
        -----
        Coles (2001) p. 55 eq. (3.7) - (3.9).

        """
        loc, scale, shape = ca.vertsplit(theta)
        znorm = (extremes - loc) / scale
        mlogs = -extremes.size1() * ca.log(scale)

        shape_zero = ca.sum1(znorm) + ca.sum1(ca.exp(-znorm))  # Gumbel limit

        shape_nonz = (
            (1. + 1. / shape)
            * ca.sum1(ca.log(1. + shape * znorm))
            + ca.sum1((1. + shape*znorm) ** (-1./shape))
        )

        return (
            mlogs
            - ca.if_else(ca.fabs(shape) < self.tiny, shape_zero, shape_nonz)
        )

    def return_level_expr(self, theta, proba):
        """Builds a Casadi expression for the GEV distribution return level.

        Parameters
        ----------
        theta : ca.MX
            Casadi symbolic placeholder for the maximum likelihood parameters.
        proba : float
            Exceedance probability (corresponding to the 1 / proba return
            level).

        Returns
        -------
        ca.MX
            Casadi symbolic return level expression.

        Notes
        -----
        The input probability `proba` is the *exceedance* probability.
        The return level with 10% exceedance probability is the
        90th percentile, which we can directly evaluate based on the
        GEV quantile function.

        """

        # Evaluate quantile at the corresponding NON-exceedance probability!
        return self.quantile(theta, 1. - proba)

    def quantile(self, theta, proba):
        """Builds a Casadi expression for the GEV distribution quantiles.

        Parameters
        ----------
        theta : ca.MX
            Casadi symbolic placeholder for the maximum likelihood parameters.
        proba : float
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

        return ca.if_else(ca.fabs(shape) < self.tiny, shape_zero, shape_nonz)

    def fit(self):
        """Fit the maximum likelihood parameters."""
        self._fit(self.extremes)

    def return_level(self, return_period):
        """Infer return level value given a return period.

        Parameters
        ----------
        return_period : float array-like
            Collection of return periods to evaluate.

        Returns
        -------
        dict
            The corresponding return level estimate and confidence interval.
        """

        return_period = np.atleast_2d(return_period)  # For casadi broadcasting
        exceedance_proba = 1. / return_period

        return self.return_level_fn(
            theta=self.theta, proba=exceedance_proba, covar=self.covar
        )

    def probability_plot(self, ax, **plot_kwargs):

        return self._probability_plot(ax, self.extremes, **plot_kwargs)

    def quantile_plot(self, ax, **plot_kwargs):

        return self._quantile_plot(ax, self.extremes, **plot_kwargs)

    def density_plot(self, ax, **plot_kwargs):

        return self._density_plot(ax, self.extremes, **plot_kwargs)
