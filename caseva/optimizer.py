"""
Maximum Likelihood Estimation base model class.
"""

import warnings
from abc import ABC, abstractmethod
import numpy as np
import casadi as ca


IPOPT_PLUGIN_OPTS = {
    "ipopt.print_level": 0,
    "print_time": 0,
    "ipopt.sb": "yes",
    "expand": True
    }

IPOPT_SOLVER_OPTS = {
    "max_iter": 1e10
    }


class MLEOptimizer(ABC):
    """Base class for a maximum likelihood estimation in casadi."""

    def __init__(
        self,
        seed,
        max_optim_restarts,
        optim_bounds,
        *args,
        **kwargs
    ):
        """Base class for a maximum likelihood estimation with Casadi.

        Parameters
        ----------
        seed : int
            Seed for randomly drawing an optimizer initial guess.
        max_optim_restarts : int
            How many randomly initialized optimizer restarts to perform if no
            solution is found on the first try.
        optimizer_bounds : np.ndarray, shape=(num_params, 2)
            Upper and lower bound for each optimized parameter.
        """
        super().__init__(*args, **kwargs)

        self.rng = np.random.default_rng(seed)
        self.max_optim_restarts = max_optim_restarts
        self.optim_bounds = optim_bounds

        # Assigned when calling `_fit`
        self.theta = None
        self.covar = None

    @property
    @abstractmethod
    def num_params(self):
        """Number of parameters to fit."""

    @abstractmethod
    def constraints_fn(self, theta, extremes):
        """Builds the constraints passed to the numerical optimizer.

        Parameters
        ----------
        theta : ca.MX
            Symbolic placeholder for the maximum likelihood parameter estimate.
        extremes : ca.DM
            Observed extreme values.

        Returns
        -------
        constr : list
            Collection of symbolic constraint expressions.
        """

    @abstractmethod
    def optimizer_initial_guess(self, extremes):
        """Calculate an initial guess value for the optimizer.

        Parameters
        ----------
        extremes : ca.DM
            Observed extreme values.

        Returns
        -------
        list
            An initial guess for each of the fitted parameters.
        """

    @abstractmethod
    def log_likelihood(self, theta, extremes):
        """Builds the objective function passed to the numerical optimizer.

        Parameters
        ----------
        theta : ca.MX
            Symbolic placeholder for the maximum likelihood parameter estimate.
        extremes : ca.DM
            Observed extreme values.

        Returns
        -------
        ca.MX
            Symbolic log likelihood expression.
        """

    def initialize_optimizer(self, extremes):
        """Builds the symbolic optimization problem.

        Arguments:
        ----------
        extremes : ca.DM
            Extreme observartions (one-dimensional).

        Returns:
        --------
        dict
            A dictionary containing the symbolic optimization problem (opti)
            and the symbolic placeholder for the optimal parameters (theta).
        """

        opti = ca.Opti()

        theta = opti.variable(self.num_params)
        opti.minimize(-self.log_likelihood(theta, extremes))

        [
            opti.subject_to(constraint)
            for constraint in self.constraints_fn(theta, extremes)
        ]

        opti.solver("ipopt", IPOPT_PLUGIN_OPTS, IPOPT_SOLVER_OPTS)

        return {"opti": opti, "theta": theta}

    def _fit(self, extremes):
        """Fit the MLE model.

        Attempts to find a solution up to 1 + self.max_optim_restarts times.

        Arguments:
        ----------
        extremes : Union[list, np.ndarray]
            The array of observed data. Must be one-dimensional.

        """

        if np.max(np.abs(extremes)) > 100.0:
            warnings.warn("Encountered large data values. Consider rescaling.")

        assert np.array(extremes).ndim == 1, "Extremes must be 1-dimensional."

        extremes = ca.DM(extremes)
        opti = self.initialize_optimizer(extremes)

        initial_guess = self.optimizer_initial_guess(extremes)
        assert len(initial_guess) == self.num_params, "Wrong size guess!"

        sol = None
        for iteration in range(1 + self.max_optim_restarts):

            if iteration > 0:
                initial_guess = self.rng.uniform(
                    self.optim_bounds[:, 0], self.optim_bounds[:, 1])

            opti["opti"].set_initial(opti["theta"], initial_guess)

            try:
                sol = opti["opti"].solve()
                assert sol.stats()["success"]
                break

            except (RuntimeError, AssertionError):
                continue

        if sol is None:
            raise ValueError("Optimization failed. No solution found.")

        self.theta = sol.value(opti["theta"])

        hessian = sol.value(ca.hessian(opti["opti"].f, opti["opti"].x)[0])
        self.covar = np.array(ca.inv(hessian))
