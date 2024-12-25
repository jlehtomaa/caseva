"""Extreme value analysis base model class."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt

from caseva.utils import ca2np, ecdf


DEFAULT_PLOT_RETURN_PERIODS = np.linspace(1.1, 1000, 500)


class BaseModel(ABC):
    """Base model class for extreme value analysis.

    Attributes
    ----------
    data : np.ndarray
        A 1d array of observed extreme values.
    num_years : int
        Indicates how many years of observations does `data` correspond to.
    theta : Optional[np.ndarray]
        Fitted parameters of the extreme value distribution. Set when calling
        the `fit` method.
    covar : Optional[np.ndarray]
        Covariance matric of the fitted extreme value parameters. Set when
        calling the `fit` method.
    dist : GenExtreme | GenPareto
        Distribution corresponding to the extreme value model.
    optimizer : MLEOptimizer
        A symbolic maximum-likelihood -based optimizer.
    """

    def __init__(self):
        """Base model class for extreme value analysis."""

        # Attributes calculated when calling the `_run_optimizer` method.
        self.theta: Optional[np.ndarray] = None
        self.covar: Optional[np.ndarray] = None

    def _run_optimizer(self, optim_bounds: np.ndarray) -> None:
        """Fit the extreme value distribution parameters with MLE.

        Parameters
        ----------
        optim_bounds : np.ndarray
            Upper and lower bounds for finding the optimal parameter vector.
            Must be of shape (n, 2), where n is the size of the parameter
            vector.
        """

        initial_guess = self.optimizer_initial_guess()

        self.theta, self.covar = self.optimizer.solve(
            data=self.data,
            objective_fn=self.log_likelihood,
            constraints_fn=self.constraints_fn,
            initial_guess=initial_guess,
            optim_bounds=optim_bounds
        )

    @staticmethod
    def _build_return_level_func(
        num_mle_params: int,
        return_level_expr: ca.MX
    ) -> ca.Function:
        """Build function for return level and uncertainty with delta method.

        Parameters
        ----------
        num_mle_params : int
            Number of maximum likelihood estimated parameters.
        return_level_expr : ca.MX
            Symbolic return level expression.

        Returns
        -------
        ca.Function
            A function evaluating return level and its confidence interval as a
            function of MLE params, exceedance probability, and covariance.

        Notes
        -----
        See Coles (2001) p.33, Chapter 2.6: Parametric Modeling.
        """

        proba = ca.MX.sym("proba", 1)
        theta = ca.MX.sym("theta", num_mle_params)
        covar = ca.MX.sym("covar", num_mle_params, num_mle_params)

        level = return_level_expr(theta, proba)
        grad = ca.jacobian(level, theta)
        error = 1.96 * ca.sqrt(grad @ covar @ grad.T)

        ca_func = ca.Function(
            "return_level_fn",
            [theta, proba, covar],
            [level, level+error, level-error],
            ["theta", "proba", "covar"],
            ["level", "upper", "lower"]
        )

        def np_func(theta, proba, covar):
            """Convert result from casadi matrices to numpy arrays."""
            res = ca_func(theta=theta, proba=proba, covar=covar)
            return {k: ca2np(v) for (k, v) in res.items()}

        return np_func

    @abstractmethod
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

    def _quantile_plot(
        self,
        ax: plt.Axes,
        **scatter_kwargs
    ) -> plt.Axes:
        """Plots modelled and empirical quantiles for each point in `data`.

        For a good fit model, the points should fall close to a 45-deg line.

        Parameters
        ----------
        ax : plt.Axes
            The matplotlib axes on which to plot the quantiles.
        scatter_kwargs
            Additional style keyword arguments for the scatter plot.

        Notes
        -----
        Coles (2001) p. 37 / p. 58.
        """

        emp_quantiles, emp_probas = ecdf(self.data)
        model_quantiles = ca2np(self.dist.quantile(self.theta, emp_probas))

        ax.scatter(model_quantiles, emp_quantiles, **scatter_kwargs)

        # Starting and ending points for the 45-deg line.
        line_start = min(model_quantiles.min(), emp_quantiles.min())
        ax.axline((line_start, line_start), slope=1)

        ax.set_ylabel("Empirical")
        ax.set_xlabel("Model")
        ax.set_title("Quantile plot")

        return ax

    def _probability_plot(self, ax: plt.Axes, **scatter_kwargs) -> plt.Axes:
        """Plots modelled and empirical distribution funtions.

        For a good fit model, the points should fall close to a 45-deg line.

        Parameters
        ----------
        ax : plt.Axes
            The matplotlib axes on which to plot the distribution function.
        scatter_kwargs
            Additional style keyword arguments for the scatter plot.

        Notes
        -----
        Coles (2001) p. 37 / p. 58.
        """

        emp_quantiles, emp_probas = ecdf(self.data)
        model_probas = self.dist.cdf(emp_quantiles, self.theta)

        ax.scatter(emp_probas, model_probas, **scatter_kwargs)
        ax.axline((0, 0), slope=1)  # 45-deg line for model evaluation.

        ax.set_ylabel("Model")
        ax.set_xlabel("Empirical")
        ax.set_title("Probability plot")

        return ax

    def empirical_return_periods(self) -> pd.Series:
        """Get empirical return periods of each input data point.

        Returns
        -------
        pd.Series
            A series where the index denotes the return periods and data the
            corresponding return values.
        """

        sorted_values = np.sort(self.data)[::-1]  # descending
        ranking = np.arange(1, len(sorted_values) + 1)

        return_periods = (self.num_years + 1) / ranking
        return pd.Series(data=sorted_values, index=return_periods).sort_index()

    def _return_level_plot(
        self,
        ax: plt.Axes,
        return_periods: Optional[List[int]] = None,
        return_level_uncertainty: bool = True,
        **scatter_kwargs
    ) -> plt.Axes:
        """Plot modelled and empirical return levels.

        Parameters
        ----------
        ax : plt.Axes
            Axis on which to plot.
        return_periods : List[int], optional, default=None
            Return periods to evaluate. If None, default values are provided.
        return_level_uncertainty : bool, default=True
            Whether to add the uncertainty range to the plot.
        scatter_kwargs
            Additional style keyword arguments for the scatter plot.
        """

        if return_periods is None:
            return_periods = DEFAULT_PLOT_RETURN_PERIODS

        emp_rp = self.empirical_return_periods()
        emp_rp = emp_rp[emp_rp.index >= 1]  # Filter tiny RPs to reduce clutter

        # Determining where to start plotting the return levels:
        # somewhere between the requested lowest return period and the
        # lowest available empirical value seems to work reasonably well.
        min_rp = (min(return_periods) + emp_rp.index.min()) / 2.
        return_periods = [rp for rp in return_periods if rp >= min_rp]

        return_levels = self.return_level(return_periods)

        ax.plot(return_periods, return_levels["level"])

        if return_level_uncertainty:
            upper = return_levels["upper"]
            lower = return_levels["lower"]
            ax.fill_between(return_periods, upper, lower, alpha=0.4)

        ax.scatter(emp_rp.index, emp_rp.values, **scatter_kwargs)

        ax.set_xscale('log')

        ax.set_xlabel("Return Period")
        ax.set_ylabel("Return Level")
        ax.set_title("Return level plot")

        return ax

    def _density_plot(self, ax: plt.Axes) -> plt.Axes:
        """Plot observed and modelled probability densities.

        Parameters
        ----------
        ax : plt.Axes
            Axis on which to plot.
        """

        ax.hist(self.data, density=True, rwidth=0.95)
        density_axis = np.linspace(self.data.min(), self.data.max(), 100)
        ax.plot(
            density_axis, self.dist.pdf(density_axis, self.theta), color="k"
        )
        ax.set_xlabel("z")
        ax.set_ylabel("f(z)")
        ax.set_title("Density plot")

    def diagnostic_plot(
        self,
        return_periods: Optional[List[int]] = None,
        return_level_uncertainty: bool = True,
    ) -> None:
        """Visual model evaluation.

        Parameters
        ----------
        return_periods : List[int], optional, default=None
            Return periods to evaluate. If None, default values are provided.
        return_level_uncertainty : bool, default=True
            Whether to add the uncertainty range to the plot.

        A subplot with:
            - probability plot
            - quantile plot
            - return levels plot
            - density plot
        """

        fig, ax = plt.subplots(2, 2)
        scatter_kwargs = {"s": 4, "color": "black", "alpha": 0.5}

        # Probabilities
        self._probability_plot(ax[0, 0], **scatter_kwargs)

        # Quantiles
        self._quantile_plot(ax[0, 1], **scatter_kwargs)

        # Return levels
        self._return_level_plot(
            ax[1, 0],
            return_periods=return_periods,
            return_level_uncertainty=return_level_uncertainty,
            **scatter_kwargs
        )

        # Densities
        self._density_plot(ax[1, 1])

        fig.tight_layout()
