"""
Implementation of the threshold excess model with a GenPareto distribution.
"""

import numpy as np
import casadi as ca


from caseva.optimizer import MLEOptimizer
from caseva.models import EVABaseModel
from caseva.common import build_return_level_func, ca2np

DEFAULT_OPTIMIZER_BOUNDS = np.array([
    [1e-8, 1000],  # Scale, \sigma
    [-1, 1000]     # Shape, \xi
    ])


class ThresholdExcessModel(MLEOptimizer, EVABaseModel):
    """
    Threshold excess model with a generalized Pareto distribution.

    Notes
    -----
    There are only two parameters for the MLE: the scale and the shape. 
    The threshold exceedance frequency (zeta) is estimated separately.
    See Coles (2001) p.82.
    """

    num_params = 2

    def __init__(self, data, threshold, num_years, max_optim_restarts=0, seed=0):
        """

        Parameters
        ----------
        data : np.ndarray
            All observed data.
        threshold : float
            Observations exceeding this high threshold are considered extreme.
        num_years : int
            How many years of observations does the data correspond to.
        max_optim_restarts : int, default=0
            How many randomly initialized optimizer restarts to perform if no
            solution is found.
        seed : int, default=0
            Seed for generating random optimizer restarts.
        """

        self.data = data
        self.threshold = threshold

        extremes = data[data > threshold]
        self.excesses = extremes - threshold

        self.return_level_fn = build_return_level_func(
            self.num_params + 1, self.return_level_expr)

        MLEOptimizer.__init__(self, seed, max_optim_restarts, DEFAULT_OPTIMIZER_BOUNDS)
        EVABaseModel.__init__(self, extremes, num_years)

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

        scale, shape = ca.vertsplit(theta)

        constr = [
            (self.optim_bounds[:, 0] <= theta) <= self.optim_bounds[:, 1],
            1. + shape * extremes / scale > 0.0
        ]

        return constr

    @staticmethod
    def optimizer_initial_guess(extremes):
        """Derive the initial guess for the MLE optimizer.

        Use the same value as in the 'ismev' R package that accompanies the
        Coles 2001 book: https://github.com/cran/ismev/blob/master/R/gpd.R

        The scale_init is based on the method of moments for Gumbel distribution.

        Arguments:
        ----------
        extremes : array-like
            The extreme observations used for maximum likelihood estimation.
        """

        scale_init = np.sqrt(6. * np.var(extremes)) / np.pi
        shape_init = 0.1
        return [scale_init, shape_init]

    def cdf(self, y):
        """GPD cumulative distribution function.

        y: threshold excess
        See Coles (2001) p.75 eq. 4.2
        """

        scale, shape = self.theta

        assert np.all(y >= 0.), "Exceedances should be greater than zero!"
        if np.abs(shape) < self.tiny:
            support_ub = -scale / shape
            assert np.all(y) <= support_ub, "Input value outside support."
            return 1. - np.exp(-y / scale)

        return 1. - (1. + (shape * y) / scale) ** (-1. / shape)


    def pdf(self, y):

        scale, shape = self.theta

        if np.abs(shape) < self.tiny:
            return (1. / scale) * np.exp(-y / scale)
        
        return (1. / scale) * (1 + (shape/scale)*y) **(-(1/shape + 1.))

    def log_likelihood(self, theta, excesses):
        """Generalized Pareto Distribution log likelihood.

        Arguments
        ---------
        theta : ca.MX
            Casadi symbolic expression for the MLE parameters.

        excesses : array-like
            The extreme observations used for maximum likelihood estimation.

        Returns
        -------
        loglik : casadi expression
            Symbolic log-likelihood.

        Notes
        -----
        Coles (2001) p.80 eq. 4.10.

        """

        scale, shape = ca.vertsplit(theta)

        xi_zero = (1. / scale) * ca.sum1(excesses)
        xi_nonz = (1. + 1. / shape) \
                * ca.sum1(ca.log(1. + shape * excesses / scale))

        loglik = -excesses.size1() * ca.log(scale) \
               - ca.if_else(ca.fabs(shape) < self.tiny, xi_zero, xi_nonz)

        return loglik


    def quantile(self, theta, proba):
        """Coles 2001 p.84

        Hosking & Wallis (1987): Parameter and Quantile Estimation for the
        Generalized Pareto Distribution 
        proba : nonexceedance probability
        (1 - proba) is the exceedance probability
        """
        scale, shape = theta

        xi_zero = -scale * ca.log(1 - proba) # should there be a minus here in from? compare hoskings p.342 vs Coles p.82
        xi_nonz = -(scale / shape) * (1 - (1 - proba) ** (-shape))

        return ca.if_else(ca.fabs(shape) < self.tiny, xi_zero, xi_nonz)


    def return_level_expr(self, theta, exp_obs):
        """
        Coles (2001)  p. 82. / p.84
        exp_crossings = N * ny * zeta = N * (n/num_years) * (k/n)
                      = N * (k / num_years)
        """

        exc_proba, scale, shape = ca.vertsplit(theta)

        # Expected number of threshold exceedances over the return period:
        exp_crossings = exc_proba * exp_obs
        nonexceedance_prob = 1 - (1 / exp_crossings)

        return self.threshold + self.quantile([scale, shape], nonexceedance_prob)

    def fit(self):
        self._fit(self.excesses)

    def get_covar(self, exc_proba):
        """
        exc_proba (zeta = k/n)


        Coles (2001) p.82, definition of V.

        """

        covar = np.zeros((3, 3))
        covar[0, 0] = exc_proba * (1 - exc_proba) / len(self.data)
        covar[1:, 1:] = self.covar

        return covar

    def return_level(self, return_period):

        return_period = np.atleast_2d(return_period) # for casadi broadcasting

        # Expected number of observations over return period:
        exp_obs = return_period * (len(self.data) / self.num_years)

        # The probability of an individual observation exceeding the
        # high threshold u (parameter `zeta` in Coles (2001)):
        exc_proba = len(self.extremes) / len(self.data) # exc_frac

        theta = np.concatenate([[exc_proba], self.theta])
        covar = self.get_covar(exc_proba=exc_proba)

        level = ca2np(self.return_level_fn(theta, exp_obs))
        error = ca2np(self.return_stder_fn(theta, exp_obs, covar)) * 1.96

        return {"level": level, "upper": level + error, "lower": level - error}
