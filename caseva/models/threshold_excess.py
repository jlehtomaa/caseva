"""
Implementation of the threshold excess model with a GenPareto distribution.
"""

from typing import List, Dict
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from caseva.optimizer import MLEOptimizer
from caseva.models import BaseModel
from caseva.common import build_return_level_func

OPTIM_BOUNDS = np.array([
    [1e-8, 50],  # Scale, \sigma
    [-1, 10]     # Shape, \xi
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
        max_optim_restarts : int, default = 0
            How many randomly initialized optimizer restarts to perform if no
            solution is found.
        seed : int, default = 0
            Seed for generating random optimizer restarts.
        """

        self.data = data
        self.threshold = threshold

        extremes = data[data > threshold]

        if extremes.size == 0:
            raise ValueError("Too high threshold, no values exceed it!")

        super().__init__(
            extremes=extremes,
            seed=seed,
            max_optim_restarts=max_optim_restarts,
            num_years=num_years,
            optim_bounds=OPTIM_BOUNDS
        )

        # Evaluate return levels with the augmented parameter vector (including
        # the threshold exceedance probability `zeta`, see Coles (2001) p. 82)
        self.return_level_fn = build_return_level_func(
            self.num_params + 1, self.return_level_expr)

        self.excesses = extremes - threshold

        # The probability of an individual observation exceeding the
        # high threshold u (parameter `zeta` in Coles (2001.)).
        self.thresh_exc_proba = len(self.excesses) / len(self.data)

    def constraints_fn(self, theta: ca.MX, extremes: ca.DM) -> List[ca.MX]:
        """Builds the log likelihood constraints for the numerical optimizer.

        Parameters
        ----------
        theta : ca.MX
            Symbolic placeholder for the maximum likelihood parameter estimate.
        extremes : ca.DM
            Observed extreme values.

        Returns
        -------
        constr : list of ca.MX
            Collection of symbolic constraint expressions.

        """

        scale, shape = ca.vertsplit(theta)

        constr = [
            (self.optim_bounds[:, 0] <= theta) <= self.optim_bounds[:, 1],
            1. + shape * extremes / scale > 1e-6
        ]

        return constr

    @staticmethod
    def optimizer_initial_guess(extremes: ca.DM) -> List[float]:
        """Derive the initial guess for the MLE optimizer.

        Use the same value as in the 'ismev' R package that accompanies the
        Coles 2001 book: https://github.com/cran/ismev/blob/master/R/gpd.R

        The scale_init is based on the method of moments for Gumbel
        distribution.

        Arguments:
        ----------
        extremes : ca.DM
            The extreme observations used for maximum likelihood estimation.
        """

        scale_init = np.sqrt(6. * np.var(extremes)) / np.pi
        shape_init = 0.1
        return [scale_init, shape_init]

    def cdf(self, x: np.ndarray) -> np.ndarray:
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

        scale, shape = self.theta

        if (x <= 0).any():
            raise ValueError("Exceedances must be strictly greater than zero.")

        if shape < 0:
            support_upper_limit = -scale / shape
            if np.any(x > support_upper_limit):
                raise ValueError("Input value outside of support.")
            return 1. - np.exp(-x / scale)

        return 1. - (1. + shape * x / scale) ** (-1. / shape)

    def pdf(self, x: np.ndarray) -> np.ndarray:
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

        scale, shape = self.theta

        if np.abs(shape) < self.tiny:
            return np.exp(-x / scale) / scale

        return ((1. + shape * x / scale) ** (-(1. / shape + 1.))) / scale

    def log_likelihood(self, theta: ca.MX, excesses: ca.DM) -> ca.MX:
        """Generalized Pareto Distribution log likelihood.

        Arguments
        ---------
        theta : ca.MX
            Casadi symbolic expression for the MLE parameters.

        excesses : ca.DM
            The extreme observations used for maximum likelihood estimation.

        Returns
        -------
        ca.MX
            Symbolic log likelihood expression.

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

    def quantile(self, theta: ca.MX, proba: ca.MX) -> ca.MX:
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
        scale, shape = theta

        shape_zero = - scale * ca.log(1 - proba)
        shape_nonz = - (scale / shape) * (1. - (1. - proba) ** (-shape))

        return ca.if_else(ca.fabs(shape) < self.tiny, shape_zero, shape_nonz)

    def return_level_expr(self, theta: ca.MX, proba: ca.MX) -> ca.MX:
        """Symbolic expression for the GPD return level (threshold + excess).

        Parameters
        ----------
        theta : ca.MX
            Casadi symbolic placeholder for the maximum likelihood parameters.
        proba : ca.MX
            Exceedance probability (corresponding to the 1 / proba return
            level).

        Returns
        -------
        ca.MX
            Casadi symbolic return level expression.

        Notes
        -----
        Coles (2001)  p. 82. / p.84
        """

        _, scale, shape = ca.vertsplit(theta)

        return self.threshold + self.quantile([scale, shape], 1 - proba)

    def fit(self) -> None:
        """Fit the maximum likelihood parameters."""
        self._fit(self.excesses)

    @property
    def augmented_covar(self) -> np.ndarray:
        """Covariance matrix of fitted params with the exceedence probability.

        Notes
        -----
        Coles (2001) p.82, definition of V.

        """

        covar = np.zeros((3, 3))

        # Binolmial variance for the probability of an individual observation
        # exceeding the high threshold.
        exc_freq_variance = (
            self.thresh_exc_proba
            * (1 - self.thresh_exc_proba) / len(self.data)
        )
        covar[0, 0] = exc_freq_variance
        covar[1:, 1:] = self.covar

        return covar

    def return_level(self, return_period: np.ndarray) -> Dict[str, np.ndarray]:
        """Infer return level values based on return periods.

        Parameters
        ----------
        return_period : np.ndarray
            Collection of return periods to evaluate.

        Returns
        -------
        dict[str, np.ndarray]
            The corresponding return level estimate and confidence interval.
            The keys are `level`, `upper`, and `lower`.
        """

        return_period = np.atleast_2d(return_period)  # for casadi broadcasting

        augmented_theta = np.concatenate([[self.thresh_exc_proba], self.theta])

        # Annualized average rate of exceeding the high threshold u.
        annual_rate_thresh_exceed = len(self.excesses) / self.num_years

        # Given that threshold exceedances happen `annual_rate_thresh_exceed`
        # times per year on average, what is the magnitude of an event that
        # would, on average, be exceeded only once in, say, 100 years?

        adj_exceed_proba = 1. / (return_period * annual_rate_thresh_exceed)

        return self.return_level_fn(
            theta=augmented_theta,
            proba=adj_exceed_proba,
            covar=self.augmented_covar
        )

    def probability_plot(self, ax: plt.Axes, **plot_kwargs) -> plt.Axes:
        """Plot empirical and modelled (cumul.) probabilities of extremes."""
        return self._probability_plot(ax, self.excesses, **plot_kwargs)

    def quantile_plot(self, ax: plt.Axes, **plot_kwargs) -> plt.Axes:
        """Plot empirical and modelled quantiles of extremes."""
        return self._quantile_plot(ax, self.excesses, **plot_kwargs)

    def density_plot(self, ax: plt.Axes, **plot_kwargs) -> plt.Axes:
        """Plot empirical and modelled probability densities of extremes."""
        return self._density_plot(ax, self.excesses, **plot_kwargs)
