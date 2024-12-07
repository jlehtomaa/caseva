"""
Implementation of the threshold excess model with a GenPareto distribution.
"""

from typing import List, Dict
import numpy as np
import casadi as ca

from caseva.models import BaseModel
from caseva.optimizer import MLEOptimizer
from caseva.distributions.genpareto import GenPareto
from caseva.utils import is_almost_zero


DEFAULT_OPTIM_BOUNDS = np.array([
    [1e-8, 50],  # Scale, \sigma
    [-1, 10]     # Shape, \xi
])


class ThresholdExcessModel(BaseModel):
    """
    Threshold excess model with a generalized Pareto distribution.

    Notes
    -----
    There are only two parameters for the MLE: the scale and the shape.
    The threshold exceedance frequency (zeta) is estimated separately.
    See Coles (2001) p.82.
    """

    def __init__(
        self,
        max_optim_restarts: int = 0,
        optim_bounds: np.ndarray = None,
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

        super().__init__()

        self.dist = GenPareto()
        self.optimizer = MLEOptimizer(
            seed=seed,
            max_optim_restarts=max_optim_restarts
        )

        if optim_bounds is None:
            optim_bounds = DEFAULT_OPTIM_BOUNDS

        self.optim_bounds = optim_bounds

        # Evaluate return levels with the augmented parameter vector (including
        # the threshold exceedance probability `zeta`, see Coles (2001) p. 82)
        self.return_level_fn = self._build_return_level_func(
            num_mle_params=self.dist.num_params + 1,
            return_level_expr=self.return_level_expr
        )

    def constraints_fn(self, theta: ca.MX, data: ca.DM) -> List[ca.MX]:
        """Builds the log likelihood constraints for the numerical optimizer.

        Parameters
        ----------
        theta : ca.MX
            Symbolic placeholder for the maximum likelihood parameter estimate.
        data : ca.DM
            Observed extreme values.

        Returns
        -------
        constr : list of ca.MX
            Collection of symbolic constraint expressions.

        """

        scale, shape = ca.vertsplit(theta)

        constr = [
            1. + shape * data / scale > 1e-6
        ]

        return constr

    def optimizer_initial_guess(self) -> np.ndarray:
        """Derive the initial guess for the MLE optimizer.

        Use the same value as in the 'ismev' R package that accompanies the
        Coles 2001 book: https://github.com/cran/ismev/blob/master/R/gpd.R

        The scale_init is based on the method of moments for Gumbel
        distribution.

        Arguments:
        ----------
        data : np.ndarray
            The extreme observations used for maximum likelihood estimation.
        """

        scale_init = np.sqrt(6. * np.var(self.data)) / np.pi
        shape_init = 0.1
        return np.array([scale_init, shape_init])

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
        shape_nonzero = (
            (1. + 1. / shape)
            * ca.sum1(ca.log(1. + shape * excesses / scale))
        )

        return (
            - excesses.size1() * ca.log(scale)
            - ca.if_else(is_almost_zero(shape), shape_zero, shape_nonzero)
        )

    def return_level_expr(self, augmented_theta: ca.MX, proba: ca.MX) -> ca.MX:
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

        # Ignore the first element in augmented_theta
        return self.dist.quantile(augmented_theta[1:], 1 - proba)

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
            * (1 - self.thresh_exc_proba) / self.num_observations
        )
        covar[0, 0] = exc_freq_variance
        covar[1:, 1:] = self.covar

        return covar

    @property
    def augmented_theta(self):
        return np.concatenate([[self.thresh_exc_proba], self.theta])

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

        # Annualized average rate of exceeding the high threshold u.
        annual_rate_thresh_exceed = len(self.data) / self.num_years

        # Given that threshold exceedances happen `annual_rate_thresh_exceed`
        # times per year on average, what is the magnitude of an event that
        # would, on average, be exceeded only once in, say, 100 years?

        adj_exceed_proba = 1. / (return_period * annual_rate_thresh_exceed)

        return self.return_level_fn(
            theta=self.augmented_theta,
            proba=adj_exceed_proba,
            covar=self.augmented_covar
        )

    def fit(self, data, threshold, num_years):

        self.threshold = threshold
        self.num_years = num_years

        self.threshold = threshold
        self.num_observations = data.size

        excesses = data[data > threshold] - threshold

        if excesses.size == 0:
            raise ValueError("Too high threshold, no values exceed it!")

        self.data = excesses

        # The probability of an individual observation exceeding the
        # high threshold u (parameter `zeta` in Coles (2001.)).
        self.thresh_exc_proba = excesses.size / self.num_observations

        self._run_optimizer()
