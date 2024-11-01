"""Extreme value analysis base model class."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from caseva.common import empirical_return_periods, ca2np


DEFAULT_PLOT_RETURN_PERIODS = np.linspace(1.1, 1000, 500)


class BaseModel(ABC):
    """Base model class for extreme value analysis.

    Attributes
    ----------
    extremes : np.ndarray
        A 1d array of observed extreme values.
    num_years : int
        Indicates how many years of observations does `extremes` correspond to.
    theta : Optional[np.ndarray]
        Fitted parameters of the extreme value distribution. Set when calling
        the `fit` method.
    covar : Optional[np.ndarray]
        Covariance matric of the fitted extreme value parameters. Set when
        calling the `fit` method.
    """

    tiny = 1e-8  # Small number for evaluating values close to zero.

    def __init__(self, extremes: np.ndarray, num_years: int, *args, **kwargs):
        """Base model class for extreme value analysis.

        Parameters
        ----------
        extremes : np.ndarray
            A 1d array of observed values.
        num_years : int
            Indicates how many years of observations does `extremes` correspond
            to. Used for evaluating return levels, not for fitting parameters.
        """
        super().__init__(*args, **kwargs)
        self.extremes = extremes
        self.num_years = num_years

        # Attributes calculated when calling the `fit` method.
        self.theta: Optional[np.ndarray] = None
        self.covar: Optional[np.ndarray] = None

    @abstractmethod
    def quantile(self, theta: ca.MX, proba: ca.MX) -> ca.MX:
        """Symbolic expression for the distribution quantile.

        Parameters
        ----------
        theta : ca.MX
            Symbolic placeholder for the maximum likelihood parameters.
        proba : ca.MX
            Non-exceedance probability.

        Returns
        -------
        ca.MX
            Symbolic quantile (non-exceedance) expression.
        """

    @abstractmethod
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function.

        Parameters
        ----------
        x : np.ndarray
            Sample quantiles to evaluate.

        Returns
        -------
        np.ndarray
            Cumulative distribution function value between 0 and 1.
        """

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function.

        Parameters
        ----------
        x : np.ndarray
            Sample points.

        Returns
        -------
        np.ndarray
            Probability density values.
        """

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

    def eval_quantile(self, prob: np.ndarray) -> np.ndarray:
        """Evaluate the symbolic quantile expression after fitting the params.

        Parameters
        ----------
        prob : np.ndarray
            The non-exceedance probabilities at which to evaluate.

        Returns
        -------
        np.ndarray
            Estimated quantile levels.

        Raises
        ------
        ValueError
            If the function is called before the model has been fitted.
        """

        if self.theta is None:
            raise ValueError("Fit the model before evaluating quantiles!")

        return ca2np(self.quantile(self.theta, prob))

    def ecdf(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Empirical cumulative distribution function.

        Parameters
        ----------
        values : np.ndarray
            Observed data values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (quantiles, probabilities): Quantiles are the sorted `values`,
            and probabilities are the empirical cumulative probabilities.
        """

        quantiles = np.sort(values)
        probabilities = np.arange(1, len(quantiles) + 1) / (len(quantiles) + 1)

        return quantiles, probabilities

    def _quantile_plot(
        self,
        ax: plt.Axes,
        extremes: np.ndarray,
        **scatter_kwargs
    ) -> plt.Axes:
        """Plots modelled and empirical quantiles for each point in `extremes`.

        For a good fit model, the points should fall close to a 45-deg line.

        Parameters
        ----------
        ax : plt.Axes
            The matplotlib axes on which to plot the quantiles.
        extremes : np.ndarray
            The observed data.
        scatter_kwargs
            Additional style keyword arguments for the scatter plot.

        Notes
        -----
        Coles (2001) p. 37 / p. 58.
        """

        emp_quantiles, emp_probas = self.ecdf(extremes)
        model_quantiles = self.eval_quantile(emp_probas)

        ax.scatter(model_quantiles, emp_quantiles, **scatter_kwargs)

        # Starting and ending points for the 45-deg line.
        line_start = min(model_quantiles.min(), emp_quantiles.min())
        ax.axline((line_start, line_start), slope=1)

        ax.set_ylabel("Empirical")
        ax.set_xlabel("Model")
        ax.set_title("Quantile plot")

        return ax

    def _probability_plot(
        self,
        ax: plt.Axes,
        extremes: np.ndarray,
        **scatter_kwargs
    ) -> plt.Axes:
        """Plots modelled and empirical distribution funtions.

        For a good fit model, the points should fall close to a 45-deg line.

        Parameters
        ----------
        ax : plt.Axes
            The matplotlib axes on which to plot the distribution function.
        extremes : np.ndarray
            The observed data.
        scatter_kwargs
            Additional style keyword arguments for the scatter plot.

        Notes
        -----
        Coles (2001) p. 37 / p. 58.
        """

        emp_quantiles, emp_probas = self.ecdf(extremes)
        model_probas = self.cdf(emp_quantiles)

        ax.scatter(emp_probas, model_probas, **scatter_kwargs)
        ax.axline((0, 0), slope=1)  # 45-deg line for model evaluation.

        ax.set_ylabel("Model")
        ax.set_xlabel("Empirical")
        ax.set_title("Probability plot")

        return ax

    def return_level_plot(
        self,
        ax: plt.Axes,
        return_periods: Optional[List[int]] = None,
        return_level_uncertainty: bool = True,
        **scatter_kwargs
    ) -> plt.Axes:
        """Plot modelled return levels.

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

        emp_rp = empirical_return_periods(self.extremes, self.num_years)
        emp_rp = emp_rp[emp_rp.index >= 1]  # Filter tiny RPs out for visuals.

        # Determining where to start plotting the return levels:
        # somewhere between the requested lowest return period and the
        # lowest available empirical value seems to work reasonably well.
        min_rp = (min(return_periods) + emp_rp.index.min()) / 2.
        return_periods = [rp for rp in return_periods if rp >= min_rp]

        return_levels = self.return_level(return_periods)

        ax.plot(return_periods, ca2np(return_levels["level"]))

        if return_level_uncertainty:
            upper = ca2np(return_levels["upper"])
            lower = ca2np(return_levels["lower"])
            ax.fill_between(return_periods, upper, lower, alpha=0.4)

        ax.scatter(emp_rp.index, emp_rp.values, **scatter_kwargs)

        ax.set_xscale('log')

        ax.set_xlabel("Return Period")
        ax.set_ylabel("Return Level")
        ax.set_title("Return level plot")

        return ax

    def _density_plot(self, ax: plt.Axes, arr: np.ndarray) -> plt.Axes:
        """Plot observed and modelled probability densities.

        Parameters
        ----------
        ax : plt.Axes
            Axis on which to plot.
        arr : float array-like
            The extreme observations to evaluate.
        """

        ax.hist(arr, density=True, rwidth=0.95)
        density_axis = np.linspace(arr.min(), arr.max(), 100)
        ax.plot(density_axis, self.pdf(density_axis), color="k")
        ax.set_xlabel("z")
        ax.set_ylabel("f(z)")
        ax.set_title("Density plot")

    def model_evaluation_plot(
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
        self.probability_plot(ax[0, 0], **scatter_kwargs)

        # Quantiles
        self.quantile_plot(ax[0, 1], **scatter_kwargs)

        # Return levels
        self.return_level_plot(
            ax[1, 0],
            return_periods=return_periods,
            return_level_uncertainty=return_level_uncertainty,
            **scatter_kwargs
        )

        # Densities
        self.density_plot(ax[1, 1])

        fig.tight_layout()
