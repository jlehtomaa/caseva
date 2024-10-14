"""
Extreme value analysis base model class.
"""

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from caseva.common import empirical_return_periods, ca2np


class EVABaseModel(ABC):

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
        prob = np.atleast_2d(prob) # for casadi broadcasting
        return ca2np(self.quantile(self.theta, prob))

    def prob_and_quantile_plot(self, ax, values, plot_type, **plot_kwargs):
        """Probability and quantile plots for model evaluation.

        Parameters
        ----------
        ax : plt.Axes
            Axis on which to plot.
        values : float array-like
            The extreme observations to evaluate.
        plot_type : str
            Which figure to plot. Must be ['quantile' or 'probability']

        Notes
        -----
        Coles (2001) p. 37.
        """

        sorted_arr = np.sort(values)
        ecdf = np.arange(1, len(sorted_arr) + 1) / (len(sorted_arr) + 1)

        if plot_type == "quantile":

            ax.plot(self.eval_quantile(ecdf), sorted_arr , "o", **plot_kwargs)

            eval_line_start = (self.eval_quantile(ecdf).min(), sorted_arr.min())
            ax.axline(eval_line_start, slope=1)

        elif plot_type == "probability":
            ax.plot(ecdf, self.cdf(sorted_arr), "o", **plot_kwargs)
            ax.axline((0,0), slope=1)

        else:
            raise ValueError("'plot_type' must be 'quantile' or 'probability'.")

        ax.set_ylabel("Empirical")
        ax.set_xlabel("Modelled")

    def return_level_plot(self, ax, return_periods):
        """Plot modelled return levels.

        Parameters
        ----------
        ax : plt.Axes
            Axis on which to plot.
        return_periods : float array-like
            Return periods to evaluate.
        """

        return_levels = self.return_level(return_periods)

        ax.plot(return_periods, ca2np(return_levels["level"]))

        upper, lower = ca2np(return_levels["upper"]), ca2np(return_levels["lower"])
        ax.fill_between(return_periods, upper, lower, alpha=0.4)

        ax.set_xscale('log')

        ax.set_xlabel("Return Period")
        ax.set_ylabel("Return Level")

    def density_plot(self, ax, values):
        """Plot observed and modelled probability densities.

        Parameters
        ----------
        ax : plt.Axes
            Axis on which to plot.
        values : float array-like
            The extreme observations to evaluate. 
        """

        ax.hist(values, density=True)
        density_axis = np.linspace(values.min(), values.max(), 100)
        ax.plot(density_axis, self.pdf(density_axis))
        ax.set_xlabel("z")
        ax.set_ylabel("f(z)")

    def model_evaluation_plot(self, **plot_kwargs):
        """Visual model evaluation.

        A subplot with:
            - probability plot
            - quantile plot
            - return levels plot
            - density plot
        """


        if self.__class__.__name__ == "ThresholdExcessModel":
            plot_vals = self.excesses
        elif self.__class__.__name__ == "BlockMaximaModel":
            plot_vals = self.extremes
        else:
            raise ValueError("Bad class name.")

        fig, ax = plt.subplots(2, 2)

        # Probability plot
        self.prob_and_quantile_plot(ax[0,0], plot_vals, "probability", **plot_kwargs)

        # Quantile plot
        self.prob_and_quantile_plot(ax[0,1], plot_vals, "quantile", **plot_kwargs)

        # Return level plot
        return_periods = np.linspace(2, 1000, 500)

        data_return_period = empirical_return_periods(self.extremes, self.num_years)
        self.return_level_plot(ax[1,0], return_periods)
        ax[1,0].plot(data_return_period, self.extremes, 'o', **plot_kwargs)

        # Density plot
        self.density_plot(ax[1,1], plot_vals)

        fig.tight_layout()
