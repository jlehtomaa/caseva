import casadi as ca
from caseva.models import BlockMaximaModel


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
