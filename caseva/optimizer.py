"""
Maximum Likelihood Estimation base model class. Sets up the Casadi optimizer.
"""

import warnings
from typing import List, Dict, Any
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
"""
Default parameters for the IPOPT solver. For all available options, see
https://coin-or.github.io/Ipopt/OPTIONS.html
"""


class MLEOptimizer(ABC):
    """Base class for a maximum likelihood estimation in casadi."""

    def __init__(
        self,
        seed: int,
        max_optim_restarts: int,
        optim_bounds: np.ndarray,
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
    def constraints_fn(self, theta: ca.MX, extremes: ca.DM) -> List[ca.MX]:
        """Builds the constraints passed to the numerical optimizer.

        Parameters
        ----------
        theta : ca.MX
            Symbolic placeholder for the maximum likelihood parameter estimate.
        extremes : ca.DM
            Observed extreme values.

        Returns
        -------
        constr : list of ca.MX
            Collection of symbolic constraint expressions.
        """

    @abstractmethod
    def optimizer_initial_guess(self, extremes: ca.DM) -> List[float]:
        """Calculate an initial guess value for the optimizer.

        Parameters
        ----------
        extremes : ca.DM
            Observed extreme values.

        Returns
        -------
        list of float
            An initial guess for each of the fitted parameters.
        """

    @abstractmethod
    def log_likelihood(self, theta: ca.MX, extremes: ca.DM) -> ca.MX:
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

    def initialize_optimizer(
        self,
        extremes: ca.DM
    ) -> Dict[str, Any]:
        """Builds the symbolic optimization problem.

        Arguments:
        ----------
        extremes : ca.DM
            Extreme observartions (one-dimensional).

        Returns:
        --------
        dict
            A dictionary containing the symbolic optimization problem
            (ca.Opti) and the symbolic placeholder for the parameters (ca.MX).
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

    @property
    def is_corner_solution(self) -> bool:
        """Check if the parameter upper and lower bounds are binding.

        This suggests a poor quality fit.

        Returns
        -------
        constraints_bind : bool
            Whether upper or lower parameter limits are binding.

        Raises
        ------
        ValueError
            If the method is called before the parameters have been fit.
        """

        if self.theta is None:
            raise ValueError(
                "`fit` must be called before checking for corner solutions."
            )

        # Check if either upper or lower bounds are binding.
        constraints_bind = any(
            np.isclose(self.theta, self.optim_bounds[:, i]).any()
            for i in range(2)
        )

        return constraints_bind

    def _fit(self, extremes: np.ndarray) -> None:
        """Fit the MLE model.

        Attempts to find a solution up to 1 + self.max_optim_restarts times.

        Arguments:
        ----------
        extremes : np.ndarray
            The array of observed data. Must be one-dimensional.

        Raises
        ------
        ValueError
            If the input array is not one-dimensional.
            If the provided initial guess is not the same size as the target
            parameters being fitted.
            If the optimization fails and no solution is found.
        """

        if np.max(np.abs(extremes)) > 100.0:
            warnings.warn(
                "Encountered large data values. Consider rescaling for better "
                "optimizer performance."
            )

        if np.array(extremes).ndim != 1:
            raise ValueError("`extremes` must be a 1-dimensional array.")

        extremes = ca.DM(extremes)  # From numpy to Casadi data matrix format.
        opti = self.initialize_optimizer(extremes)

        initial_guess = self.optimizer_initial_guess(extremes)
        if len(initial_guess) != self.num_params:
            raise ValueError(
                f"Wrong size initial guess. Got {len(initial_guess)} "
                f"but expected {self.num_params}."
            )

        sol = None
        for iteration in range(1 + self.max_optim_restarts):

            if iteration > 0:
                initial_guess = self.rng.uniform(
                    self.optim_bounds[:, 0], self.optim_bounds[:, 1]
                )

            opti["opti"].set_initial(opti["theta"], initial_guess)

            try:
                sol = opti["opti"].solve()
                assert sol.stats()["success"]
                break

            except (RuntimeError, AssertionError):
                continue

        if sol is None:
            raise ValueError("Optimization failed. No solution found.")

        hessian = sol.value(ca.hessian(opti["opti"].f, opti["opti"].x)[0])
        self.theta = sol.value(opti["theta"])
        self.covar = np.array(ca.inv(hessian))

        if self.is_corner_solution:
            warnings.warn("Corner solution encountered!")
