"""
Maximum Likelihood Estimator class. Sets up the Casadi optimizer.
"""

import warnings
from typing import Tuple, Dict, Any, Callable
import numpy as np
import casadi as ca


IPOPT_PLUGIN_OPTS = {
    "ipopt.print_level": 0,
    "print_time": 0,
    "ipopt.sb": "yes",
    "expand": True,
    "verbose": False
    }

IPOPT_SOLVER_OPTS = {
    "max_iter": 10_000
    }
"""
Default parameters for the IPOPT solver. For all available options, see
https://coin-or.github.io/Ipopt/OPTIONS.html
"""


class MLEOptimizer():
    """Class for a maximum likelihood estimation in casadi."""

    def __init__(
        self,
        seed: int,
        max_optim_restarts: int,
        result_type: str = "first"
    ):
        """Class for a maximum likelihood estimation with Casadi.

        Parameters
        ----------
        seed : int
            Seed for randomly drawing an optimizer initial guess.
        max_optim_restarts : int
            How many randomly initialized optimizer restarts to perform if no
            solution is found on the first try.
        result_type : str, default=first
            When `max_optim_restarts` is > 0, the solver can either return the
            first solution that was found with `result_type` set to "first",
            or the one with the highest likelihood by setting this variable
            to "best".
        """

        self.rng = np.random.default_rng(seed)
        self.max_optim_restarts = max_optim_restarts
        self.result_type = result_type

        # Determined during the `solve` method.
        self.is_corner_solution = None

    def solve(
        self,
        data: np.ndarray,
        objective_fn: Callable,
        constraints_fn: Callable,
        initial_guess: np.ndarray,
        optim_bounds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the MLE model.

        Parameters
        ----------
        data : np.ndarray
            The array of observed extreme value data. Must be one-dimensional.
        objective_fn : Callable
            The log likelihood function to maximize.
        constraints_fn : Callable
            Returns a list of constraints to the solver (not including the
            constraints for parameter upper and lower bounds).
        initial_guess : np.ndarray
            Initial guess of the optimal parameter vector.
        optim_bounds : np.ndarray
            Upper and lower bounds for finding the optimal parameter vector.
            Must be of shape (n, 2), where n is the size of the parameter
            vector.

        Returns
        -------
        theta : np.ndarray
            The optimal parameter vector.
        covar : np.ndarray
            The variance-covariance matrix of the fitted paramete vector.

        Raises
        ------
        ValueError
            If the optimization fails and no solution is found.
        """

        self._validate_inputs(data, initial_guess, optim_bounds)
        opti = self._initialize_solver(
            data, objective_fn, constraints_fn, initial_guess, optim_bounds)

        best_sol = None
        best_neg_log_lik = np.inf
        for iteration in range(1 + self.max_optim_restarts):

            if iteration > 0:
                # When doing multiple optimizer restarts, take a random
                # initial guess within the allowed upper and lower bounds.
                initial_guess = self.rng.uniform(
                    optim_bounds[:, 0], optim_bounds[:, 1]
                )

                opti["opti"].set_initial(opti["theta"], initial_guess)

            try:
                sol = opti["opti"].solve()
                assert sol.stats()["success"]

                neg_log_lik = sol.value(opti["opti"].f)

                # Note: We minimize the negative log likelihood.
                if neg_log_lik < best_neg_log_lik:
                    best_neg_log_lik = neg_log_lik
                    best_sol = sol

                if self.result_type == "first":
                    break

            except (RuntimeError, AssertionError):
                continue

        if best_sol is None:
            raise ValueError("Optimization failed. No solution found.")

        # Get the approximate information matrix.
        hessian = best_sol.value(ca.hessian(opti["opti"].f, opti["opti"].x)[0])

        theta = best_sol.value(opti["theta"])
        covar = np.array(ca.inv(hessian))

        if self._is_corner_solution(theta, optim_bounds):
            warnings.warn("Corner solution encountered!")
            self.is_corner_solution = True
        else:
            self.is_corner_solution = False

        return theta, covar

    @staticmethod
    def _validate_inputs(
        data: np.ndarray,
        initial_guess: np.ndarray,
        optim_bounds: np.ndarray
    ) -> None:
        """Check that the inputs for the optimizer are valid.

        Parameters
        ----------
        data : np.ndarray
            The array of observed extreme value data. Must be one-dimensional.
        initial_guess : np.ndarray
            Initial guess of the optimal parameter vector.
        optim_bounds : np.ndarray
            Upper and lower bounds for finding the optimal parameter vector.
            Must be of shape (n, 2), where n is the size of the parameter
            vector.

        Raises
        ------
        ValueError
            If the `data` array is not 1-dimensional.
            If the shape of `optim_bounds` does not match the shape of
            `initial_guess`.
            If any of the lower bounds in `optim_bounds` is strictly larger
            than the corresponding upper bound.
        """

        if np.array(data).ndim != 1:
            raise ValueError("`data` must be a 1-dimensional array.")

        if np.max(np.abs(data)) > 100.0:
            warnings.warn(
                "Encountered large absolute data values. Consider rescaling "
                "for better optimizer performance."
            )

        if optim_bounds.shape != (initial_guess.size, 2):
            raise ValueError(
                "The optimizer bounds shape must be exactly (n, 2), where n "
                "is the size of the parameter vector `theta`."
            )

        if np.any(optim_bounds[:, 0] > optim_bounds[:, 1]):
            raise ValueError(
                "Optimizer lower bound cannot be larger than the lower bound."
            )

    @staticmethod
    def _initialize_solver(
        data,
        objective_fn,
        constraints_fn,
        initial_guess,
        optim_bounds
    ) -> Dict[str, Any]:
        """Builds the symbolic optimization problem.

        Parameters
        ----------
        data : np.ndarray
            The array of observed extreme value data. Must be one-dimensional.
        objective_fn : Callable
            The log likelihood function to maximize.
        constraints_fn : Callable
            Returns a list of constraints to the solver (not including the
            constraints for parameter upper and lower bounds).
        initial_guess : np.ndarray
            Initial guess of the optimal parameter vector.
        optim_bounds : np.ndarray
            Upper and lower bounds for finding the optimal parameter vector.
            Must be of shape (n, 2), where n is the size of the parameter
            vector.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing the symbolic optimization problem
            (ca.Opti) and the symbolic placeholder for the parameters (ca.MX).
        """

        data = ca.DM(data)  # From numpy to Casadi data matrix format.

        opti = ca.Opti()
        theta = opti.variable(len(initial_guess))

        opti.minimize(-objective_fn(theta, data))

        [
            opti.subject_to(constraint)
            for constraint in constraints_fn(theta, data)
        ]

        opti.subject_to((optim_bounds[:, 0] <= theta) <= optim_bounds[:, 1])

        opti.set_initial(theta, initial_guess)
        opti.solver("ipopt", IPOPT_PLUGIN_OPTS, IPOPT_SOLVER_OPTS)

        return {"opti": opti, "theta": theta}

    @staticmethod
    def _is_corner_solution(
        theta: np.ndarray,
        optim_bounds: np.ndarray
    ) -> bool:
        """Check if the parameter upper and lower bounds are binding.
        This suggests a poor quality fit.

        Parameters
        ----------
        theta : np.ndarray
            Distribution parameters being fitted.
        optim_bounds : np.ndarray
            Upper and lower bounds for theta. Must have shape n x 2, where
            n is the size of `theta`.

        Returns
        -------
        bool
            Whether any upper or lower parameter bounds are binding.
        """

        return np.any(np.isclose(theta[:, None], optim_bounds))
