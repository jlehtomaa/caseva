"""
Implementation of the classical extreme value model with (annual) block maxima.
"""

import numpy as np
import casadi as ca

from caseva.optimizer import MLEOptimizer
from caseva.models import EVABaseModel
from caseva.common import build_return_level_func


DEFAULT_OPTIM_BOUNDS = np.array([
    [-100, 100],  # Location, \mu
    [1e-8, 100],  # Scale, \sigma
    [-1, 100]])   # Shape, \xi


class BlockMaximaModel(MLEOptimizer, EVABaseModel):
    """Classical extreme value model with annual block maxima."""

    num_params = 3

    def __init__(self, extremes, num_years, max_optim_restarts=0, seed=0):
        """

        Parameters
        ----------
        extremes : np.ndarray
            Extreme observations (annual maxima).
        num_years : int
            How many years of observations does the data correspond to.
            Only used for plotting and model evaluation.
        max_optim_restarts : int, default=0
            How many randomly initialized optimizer restarts to perform if no
            solution is found.
        seed : int, default=0
            Seed for generating random optimizer restarts.
        """

        MLEOptimizer.__init__(self, seed, max_optim_restarts, DEFAULT_OPTIM_BOUNDS)
        EVABaseModel.__init__(self, extremes, num_years)

        self.return_level_fn = build_return_level_func(
            self.num_params, self.return_level_expr)

    def constraints_fn(self, theta, extremes):
        """Builds the constraints passed to the numerical optimizer.

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

        The scale_init is based on the method of moments for Gumbel distribution.

        Parameters
        ----------
        extremes : array-like
            The extreme observations used for maximum likelihood estimation.
        """
        
        scale_init = np.sqrt(6. * np.var(extremes)) / np.pi
        loc_init =  np.mean(extremes) - 0.57722 * scale_init
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
        Coles (2001) p. 55 eq. 3.7 - 3.9.

        """
        loc, scale, shape = ca.vertsplit(theta)
        znorm = (extremes - loc) / scale
        mlogs = -extremes.size1() * ca.log(scale)

        shape_zero = ca.sum1(znorm) + ca.sum1(ca.exp(-znorm))

        shape_nonz = (1. + 1./shape) * ca.sum1(ca.log(1. + shape*znorm)) \
                   + ca.sum1((1. + shape*znorm) ** (-1./shape))

        return mlogs - ca.if_else(ca.fabs(shape) < self.tiny, shape_zero, shape_nonz)

    def return_level_expr(self, theta, proba):
        """Builds a Casadi expression for the GEV distribution return level.

        Parameters
        ----------
        theta : ca.MX
            Casadi symbolic placeholder for the maximum likelihood parameters.
        proba : float
            Non-exceedance probability.

        Returns
        -------
        ca.MX
            Casadi symbolic quantile expression.

        """
        return self.quantile(theta, 1 - proba)

    def quantile(self, theta, proba):
        """Builds a Casadi expression for the GEV distribution quantile.

        Parameters
        ----------
        theta : ca.MX
            Casadi symbolic placeholder for the maximum likelihood parameters.
        proba : float
            Non-exceedance probability.

        Returns
        -------
        ca.MX
            Casadi symbolic quantile expression.

        Notes
        -----
        For details, see Coles (2001) p. 56 eq. 3.10. / Coles 2001 p.49 eq. 3.4
        """

        logp = -ca.log(proba)
        loc, scale, shape = ca.vertsplit(theta)

        shape_zero = loc - scale * ca.log(logp)
        shape_nonz = loc - (scale / shape) * (1. - logp ** (-shape))

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

        return_period = np.atleast_2d(return_period) # For casadi broadcasting
        proba = 1. / return_period

        return self.return_level_fn(
            theta=self.theta, proba=proba, covar=self.covar)
