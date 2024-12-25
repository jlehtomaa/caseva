"""
Implementation of the classical extreme value model with (annual) block maxima.
"""
from typing import List, Dict, Optional
import numpy as np
import casadi as ca

from caseva.models import BaseModel
from caseva.optimizer import MLEOptimizer
from caseva.distributions.genextreme import GenExtreme
from caseva.utils import is_almost_zero


DEFAULT_OPTIM_BOUNDS = np.array([
    [-100, 50],  # Location, \mu
    [1e-8, 50],  # Scale, \sigma
    [-1, 10]     # Shape, \xi
])
"""
See discussion in Coles (2001) p. 55 for the shape parameter constraints.
"""


class BlockMaximaModel(BaseModel):
    """Classical extreme value model with annual block maxima."""

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

        self.dist = GenExtreme()
        self.optimizer = MLEOptimizer(
            seed=seed,
            max_optim_restarts=max_optim_restarts,
            result_type=result_type
        )

        self.return_level_fn = self._build_return_level_func(
            num_mle_params=self.dist.num_params,
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

        Notes
        -----
        Coles (2001) p. 55 eq. (3.8). The parameter constraint must hold for
        all elements in `extremes`.
        """

        loc, scale, shape = ca.vertsplit(theta)
        constr = [1. + shape * ((data - loc) / scale) > 0.]

        return constr

    def optimizer_initial_guess(self) -> np.ndarray:
        """Derive the initial guess for the MLE optimization.

        Use the same value as in the 'ismev' R package that accompanies the
        Coles 2001 book: https://github.com/cran/ismev/blob/master/R/gev.R

        The scale_init is based on the method of moments for a Gumbel
        distribution.

        Returns
        -------
        np.ndarray
            An initial guess for each of the fitted parameters.
        """

        scale_init = np.sqrt(6. * np.var(self.data)) / np.pi
        loc_init = np.mean(self.data) - 0.57722 * scale_init
        shape_init = 0.1

        return np.array([loc_init, scale_init, shape_init])

    def log_likelihood(self, theta: ca.MX, extremes: ca.DM) -> ca.MX:
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

        shape_nonzero = (
            (1. + 1. / shape)
            * ca.sum1(ca.log(1. + shape * znorm))
            + ca.sum1((1. + shape*znorm) ** (-1./shape))
        )

        return (
            mlogs
            - ca.if_else(is_almost_zero(shape), shape_zero, shape_nonzero)
        )

    def return_level_expr(self, theta: ca.MX, proba: ca.MX) -> ca.MX:
        """Symbolic expression for the GEV distribution return level.

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
        The input probability `proba` is the *exceedance* probability.
        The return level with 10% exceedance probability is the
        90th percentile, which we can directly evaluate based on the
        GEV quantile function.
        """

        return self.dist.quantile(theta, 1. - proba)

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

        return_period = np.atleast_2d(return_period)  # For casadi broadcasting
        exceedance_proba = 1. / return_period

        return self.return_level_fn(
            theta=self.theta, proba=exceedance_proba, covar=self.covar
        )

    def fit(
        self,
        data: np.ndarray,
        optim_bounds: Optional[np.ndarray] = None
    ) -> None:
        """Fit the extreme value distribution to the data.

        Parameters
        ----------
        data : np.ndarray
            A 1d array of observed values.
        optim_bounds : np.ndarray
            Upper and lower bounds for finding the optimal parameter vector.
            Must be of shape (n, 2), where n is the size of the parameter
            vector.
        """

        self.data = data
        self.num_years = data.size

        if optim_bounds is None:
            optim_bounds = DEFAULT_OPTIM_BOUNDS

        self._run_optimizer(optim_bounds=optim_bounds)
