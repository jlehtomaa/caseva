"""
Extreme value analysis base model class.
"""

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from caseva.common import empirical_return_periods, ca2np


class BaseModel(ABC):

    tiny = 1e-8

    def __init__(self, extremes, num_years):

        self.theta = None
        self.extremes = extremes
        self.num_years = num_years

    @abstractmethod
    def quantile(self, theta, proba):
        """Builds a Casadi expression for the associated distribution quantile.

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

    @abstractmethod
    def cdf(self, x):
        """Cumulative distribution function for the associated distribution.

        Parameters
        ----------
        x : Union[float, np.ndarray]
            Sample point.
        """

    @abstractmethod
    def pdf(self, x):
        """Probability density function for the associated distribution.

        Parameters
        ----------
        x : Union[float, np.ndarray]
            Sample point.
        """

    @abstractmethod
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

    def eval_quantile(self, prob):
        """Evaluate the symbolic quantile expression after fitting the params.

        Parameters
        ----------
        prob : float array-like
            The non-exceedance probabilities at which to evaluate.

        Returns
        -------
        np.ndarray
            Estimated quantile levels.
        """
        prob = np.atleast_2d(prob)  # for casadi broadcasting
        return ca2np(self.quantile(self.theta, prob))

    def ecdf(self, extremes):

        quantiles = np.sort(extremes)
        probabilities = np.arange(1, len(quantiles) + 1) / (len(quantiles) + 1)

        return quantiles, probabilities

    def _quantile_plot(self, ax, extremes, **plot_kwargs):
        """Plots modelled and empirical quantiles for each point in `extremes`.

        For a good fit model, the points should fall close to a 45-deg line.

        Notes
        -----
        Coles (2001) p. 37 / p. 58.
        """

        emp_quantiles, emp_probas = self.ecdf(extremes)
        model_quantiles = self.eval_quantile(emp_probas)

        ax.scatter(model_quantiles, emp_quantiles, **plot_kwargs)

        # Starting and ending points for the 45-deg line.
        line_start = min(model_quantiles.min(), emp_quantiles.min())
        ax.axline((line_start, line_start), slope=1)

        ax.set_ylabel("Empirical")
        ax.set_xlabel("Model")
        ax.set_title("Quantile plot")

        return ax

    def _probability_plot(self, ax, extremes, **plot_kwargs):
        """

        Notes
        -----
        Coles (2001) p. 37 / p. 58.
        """

        emp_quantiles, emp_probas = self.ecdf(extremes)
        model_probas = self.cdf(emp_quantiles)

        ax.scatter(emp_probas, model_probas, **plot_kwargs)
        ax.axline((0, 0), slope=1)  # 45-deg line for model evaluation.

        ax.set_ylabel("Model")
        ax.set_xlabel("Empirical")
        ax.set_title("Probability plot")

        return ax

    def return_level_plot(self, ax, **plot_kwargs):
        """Plot modelled return levels.

        Parameters
        ----------
        ax : plt.Axes
            Axis on which to plot.
        return_periods : float array-like
            Return periods to evaluate.
        """

        return_periods = np.linspace(1, 1000, 500)
        return_levels = self.return_level(return_periods)

        ax.plot(return_periods, ca2np(return_levels["level"]))

        upper = ca2np(return_levels["upper"])
        lower = ca2np(return_levels["lower"])
        ax.fill_between(return_periods, upper, lower, alpha=0.4)

        emp_rp = empirical_return_periods(self.extremes, self.num_years)
        ax.scatter(emp_rp, self.extremes, **plot_kwargs)

        ax.set_xscale('log')

        ax.set_xlabel("Return Period")
        ax.set_ylabel("Return Level")
        ax.set_title("Return level plot")

        ticks_and_labels = [1, 10, 100, 1000]
        ax.set_xticks(ticks_and_labels)
        ax.set_xticklabels(ticks_and_labels)

    def _density_plot(self, ax, values):
        """Plot observed and modelled probability densities.

        Parameters
        ----------
        ax : plt.Axes
            Axis on which to plot.
        values : float array-like
            The extreme observations to evaluate.
        """

        ax.hist(values, density=True, rwidth=0.95)
        density_axis = np.linspace(values.min(), values.max(), 100)
        ax.plot(density_axis, self.pdf(density_axis), color="k")
        ax.set_xlabel("z")
        ax.set_ylabel("f(z)")
        ax.set_title("Density plot")

    def model_evaluation_plot(self):
        """Visual model evaluation.

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
        self.return_level_plot(ax[1, 0], **scatter_kwargs)

        # Densities
        self.density_plot(ax[1, 1])

        fig.tight_layout()
