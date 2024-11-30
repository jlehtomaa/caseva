import numpy as np
import casadi as ca
from caseva.models import BlockMaximaModel, ThresholdExcessModel
from caseva.common import ca2np


class PointProcesModel(BlockMaximaModel):

    def __init__(
        self,
        data,
        threshold,
        num_years,
        max_optim_restarts=0,
        seed=0
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

        self.threshold = threshold
        extremes = data[data > threshold]

        self.excesses = extremes - threshold

        super().__init__(extremes, num_years, max_optim_restarts, seed)

    def log_likelihood(self, theta, excesses):
        """Non-homogeneous Poisson process log likelihood.

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
        Coles (2001) p. 134 eq. (7.9).
        """

        loc, scale, shape = ca.vertsplit(theta)
        mlogs = -excesses.size1() * ca.log(scale)
        tnorm = (self.threshold - loc) / scale
        enorm = (excesses - loc) / scale

        shape_zero = mlogs - self.num_years * ca.exp(-tnorm) - ca.sum1(enorm)

        shape_nonz = mlogs - self.num_years * (1 + shape * tnorm) ** (-1/shape) \
                   - (1+1/shape) * ca.sum1(ca.log(1 + shape*enorm))

        return ca.if_else(ca.fabs(shape) < self.tiny, shape_zero, shape_nonz)

    def cdf(self, z):

        loc, scale, shape = self.theta

        # STEP 1: PARAMETER SCALING.
        # new_loc = 1 - np.exp(
        #     ((1 + shape * (self.threshold - loc) / scale) ** (-1/shape))
        #     / self.num_years
        # )
        new_scale = scale + shape * (self.threshold - loc)


        if (z <= 0).any():
            raise ValueError("Exceedances must be strictly greater than zero.")

        if shape < 0:
            support_upper_limit = -new_scale / shape
            if np.any(z > support_upper_limit):
                raise ValueError("Input value outside of support.")
            return 1. - np.exp(-(z - self.threshold) / new_scale)

        return 1. - (1. + shape * (z - self.threshold) / new_scale) ** (-1. / shape)
    

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

        loc, scale, shape = self.theta

        # SCALE THE PARAMETERS

        new_scale = scale + shape * (self.threshold - loc)

        if np.abs(shape) < self.tiny:
            return np.exp(-x / new_scale) / new_scale

        # GPD PDF with x replaced by x-threshold
        return ((1. + shape * (x - self.threshold) / new_scale) ** (-(1. / shape + 1.))) / new_scale

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

        loc, scale, shape = self.theta

        # Parameter adjustments to link the GEV parameters to the threshold
        # exceedance framework.

        # Compute the survival function from the CDF of the GEV model to
        # get the threshold exceedance probability.
        # Then divide by the number of years to consider the rate of
        # exceedances relative to the GEV block size.

        # The GEV distribution assumes that in CDF(z), z represents block
        # maxima (e.g. annual maxima) and the shape, scale, and loc parameters
        # describe the block-level distribution. If you observe N years of
        # data, each block corresponds to one year.

        # # STEP 1: PARAMETER SCALING.
        # new_loc = 1 - np.exp(
        #     ((1 + shape * (self.threshold - loc) / scale) ** (-1/shape))
        #     / self.num_years
        # )
        new_scale = scale + shape * (self.threshold - loc)

        # STEP 2: USE GPD QUANTILE FUNCTION.
        shape_zero = - new_scale * ca.log(1 - prob)
        shape_nonz = - (new_scale / shape) * (1. - (1. - prob) ** (-shape))

        return self.threshold + ca2np(ca.if_else(ca.fabs(shape) < self.tiny, shape_zero, shape_nonz))
