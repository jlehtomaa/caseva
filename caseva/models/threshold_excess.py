"""
Implementation of the threshold excess model with a GenPareto distribution.
"""

import numpy as np
import casadi as ca


from caseva.optimizer import MLEOptimizer
from caseva.models import BaseModel
from caseva.common import build_return_level_func

DEFAULT_OPTIMIZER_BOUNDS = np.array([
    [1e-8, 100],  # Scale, \sigma
    [-1, 100]     # Shape, \xi
    ])


class ThresholdExcessModel(MLEOptimizer, BaseModel):
    """
    Threshold excess model with a generalized Pareto distribution.

    Notes
    -----
    There are only two parameters for the MLE: the scale and the shape.
    The threshold exceedance frequency (zeta) is estimated separately.
    See Coles (2001) p.82.
    """

    num_params = 2

    def __init__(
        self,
        data: np.ndarray,
        threshold: float,
        num_years: int,
        max_optim_restarts: int = 0,
        seed: int = 0
    ):
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

        if extremes.size == 0:
            raise ValueError("Too high threshold, no values exceed it!")
        self.excesses = extremes - threshold

        # Evaluate return levels with the augmented parameter vector (including
        # the threshold exceedance probability `zeta`, see Coles (2001) p. 82)
        self.return_level_fn = build_return_level_func(
            self.num_params + 1, self.return_level_expr)

        MLEOptimizer.__init__(
            self, seed, max_optim_restarts, DEFAULT_OPTIMIZER_BOUNDS
        )
        BaseModel.__init__(self, extremes=extremes, num_years=num_years)

        # The probability of an individual observation exceeding the
        # high threshold u (parameter `zeta` in Coles (2001.)).
        self.thresh_exc_proba = len(self.excesses) / len(self.data)

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
            1. + shape * extremes / scale > 1e-6
        ]

        return constr

    @staticmethod
    def optimizer_initial_guess(extremes):
        """Derive the initial guess for the MLE optimizer.

        Use the same value as in the 'ismev' R package that accompanies the
        Coles 2001 book: https://github.com/cran/ismev/blob/master/R/gpd.R

        The scale_init is based on the method of moments for Gumbel
        distribution.

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
        See Coles (2001) p.75 eq. (4.2) and p.76 eq. (4.4)
        """

        scale, shape = self.theta

        if (y <= 0).any():
            raise ValueError("Exceedances must be strictly greater than zero.")

        if shape < 0:
            support_ub = -scale / shape
            assert np.all(y <= support_ub), "Input value outside of support."
            return 1. - np.exp(-y / scale)

        return 1. - (1. + shape * y / scale) ** (-1. / shape)

    def pdf(self, y):
        """Hoskins p. 339 eq (2)."""

        scale, shape = self.theta

        if np.abs(shape) < self.tiny:
            return np.exp(-y / scale) / scale

        return ((1. + shape * y / scale) ** (-(1. / shape + 1.))) / scale

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

        shape_zero = ca.sum1(excesses) / scale
        shape_nonz = (
            (1. + 1. / shape)
            * ca.sum1(ca.log(1. + shape * excesses / scale))
        )

        return (
            - excesses.size1() * ca.log(scale)
            - ca.if_else(ca.fabs(shape) < self.tiny, shape_zero, shape_nonz)
        )

    def quantile(self, theta, proba):
        """Coles 2001 p.84

        Hosking & Wallis (1987): Parameter and Quantile Estimation for the
        Generalized Pareto Distribution
        proba is the usual non-exceedance probability
        """
        scale, shape = theta

        shape_zero = - scale * ca.log(1 - proba)
        shape_nonz = - (scale / shape) * (1. - (1. - proba) ** (-shape))

        return ca.if_else(ca.fabs(shape) < self.tiny, shape_zero, shape_nonz)

    def return_level_expr(self, theta, proba):
        """
        Coles (2001)  p. 82. / p.84
        proba: exceedance proba
        """

        _, scale, shape = ca.vertsplit(theta)
        nonexceed_prob = 1 - proba
        print(type([scale, shape]), type(scale), type(shape))
        return self.threshold + self.quantile([scale, shape], nonexceed_prob)

    def fit(self):
        self._fit(self.excesses)

    @property
    def augmented_covar(self):
        """
        exc_proba (zeta = k/n)


        Coles (2001) p.82, definition of V.

        """

        covar = np.zeros((3, 3))
        zeta = self.thresh_exc_proba
        covar[0, 0] = zeta * (1 - zeta) / len(self.data)  # Binolmial variance
        covar[1:, 1:] = self.covar

        return covar

    def return_level(self, return_period):

        return_period = np.atleast_2d(return_period)  # for casadi broadcasting

        theta = np.concatenate([[self.thresh_exc_proba], self.theta])

        # Annualized average rate of exceeding the high threshold u.
        avg_num_thresh_exceed = len(self.excesses) / self.num_years

        # Given that threshold exceedances happen `avg_num_thresh_exceed`
        # times per year on average, what is the magnitude of an event that
        # would, on average, be exceeded only once in, say, 100 years?

        adj_exceed_proba = 1. / (return_period * avg_num_thresh_exceed)

        return self.return_level_fn(
            theta=theta, proba=adj_exceed_proba, covar=self.augmented_covar
        )

    def probability_plot(self, ax, **plot_kwargs):

        return self._probability_plot(ax, self.excesses, **plot_kwargs)

    def quantile_plot(self, ax, **plot_kwargs):

        return self._quantile_plot(ax, self.excesses, **plot_kwargs)

    def density_plot(self, ax, **plot_kwargs):

        return self._density_plot(ax, self.excesses, **plot_kwargs)
