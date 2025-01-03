"""
Implementation of the threshold excess model with a GenPareto distribution.
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
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
        seed: int = 0,
        max_optim_restarts: int = 0,
        result_type: str = "first"
    ):
        """

        Parameters
        ----------
        seed : int, default = 0
            Seed for generating random optimizer restarts.
        max_optim_restarts : int, default = 0
            How many randomly initialized optimizer restarts to perform if no
            solution is found.
        result_type : str, default="first"
            When `max_optim_restarts` is > 0, the solver can either return the
            first solution that was found with `result_type` set to "first",
            or the one with the highest likelihood by setting this variable
            to "best". The latter can provide better parameter estimates when
            the optimization space is non-convex, although with a computational
            cost.
        """

        super().__init__()

        self.dist = GenPareto()
        self.optimizer = MLEOptimizer(
            seed=seed,
            max_optim_restarts=max_optim_restarts,
            result_type=result_type
        )

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

        constr = [1. + shape * data / scale > 0.]

        return constr

    def optimizer_initial_guess(self) -> np.ndarray:
        """Derive the initial guess for the MLE optimizer.

        Use the same value as in the 'ismev' R package that accompanies the
        Coles 2001 book: https://github.com/cran/ismev/blob/master/R/gpd.R

        The scale_init is based on the method of moments for Gumbel
        distribution.
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
        augmented_theta : ca.MX
            Casadi symbolic placeholder for the maximum likelihood parameters,
            augmented with the threshold exceedance probability `zeta`.
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
        """Covariance matrix of fitted params and the exceedence probability.

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
        """The Genpareto parameters with the threshold exceedance probability.
        """
        return np.concatenate([[self.thresh_exc_proba], self.theta])

    def return_level(self, return_period: np.ndarray) -> Dict[str, np.ndarray]:
        """Return level values (excess + thresh.) based on return periods.

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

        excess_levels = self.return_level_fn(
            theta=self.augmented_theta,
            proba=adj_exceed_proba,
            covar=self.augmented_covar
        )

        # Add in the threshold to model the actual event values, not only the
        # threshold exceedance levels.
        return {k: v + self.threshold for (k, v) in excess_levels.items()}

    def fit(
        self,
        data: np.ndarray,
        threshold: float,
        num_years: int,
        optim_bounds: Optional[np.ndarray] = None
    ):
        """
        Parameters
        ----------
        data : np.ndarray
            A 1d array of observed values.
        threshold : float
            The high threshold above which the generalized Pareto distribution
            is considered a valid approximation of the process.
        num_years : int
            Indicates how many years of observations does `data` correspond
            to. Used for evaluating return levels, not for fitting parameters.
        optim_bounds : np.ndarray
            Upper and lower bounds for finding the optimal parameter vector.
            Must be of shape (n, 2), where n is the size of the parameter
            vector.
        """

        self.threshold = threshold
        self.num_years = num_years

        self.num_observations = data.size

        excesses = data[data > threshold] - threshold

        if excesses.size == 0:
            raise ValueError("Too high threshold, no values exceed it!")

        self.data = excesses

        # The probability of an individual observation exceeding the
        # high threshold u (parameter `zeta` in Coles (2001.)).
        self.thresh_exc_proba = excesses.size / self.num_observations

        if optim_bounds is None:
            optim_bounds = DEFAULT_OPTIM_BOUNDS

        self._run_optimizer(optim_bounds=optim_bounds)

    def empirical_return_periods(self) -> pd.Series:
        """Adjust the empirical return levels by threshold size.
        In practice, we are interested in the value of the actual events,
        not the value of the threshold exceedance.
        """
        return super().empirical_return_periods() + self.threshold
