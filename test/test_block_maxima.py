
import pytest
import pandas as pd
import numpy as np
from caseva.models import BlockMaximaModel


@pytest.fixture
def fitted_model():
    """Block maxima model fit to the Port Pirie sea level data (Coles 2001)."""

    data = pd.read_csv("./data/portpirie.csv")
    extremes = data["SeaLevel"]
    num_years = data["Year"].max() - data["Year"].min() + 1

    model = BlockMaximaModel(extremes=extremes, num_years=num_years)
    model.fit()

    return model


def test_coles_example_3_4_1_theta(fitted_model):
    """
    Coles (2001) p. 59 example 3.4.1: Model parameters test.
    """

    fit_theta = np.round(fitted_model.theta, 3)
    expected_theta = np.array([3.875, 0.198, -0.05])
    assert np.isclose(fit_theta, expected_theta).all()


def test_coles_example_3_4_1_covar(fitted_model):
    """
    Coles (2001) p. 59 example 3.4.1: Model parameter covariance test.
    """

    fit_covar = np.round(np.diag(fitted_model.covar), 5)
    expected_covar = np.array([0.00078, 0.00041, 0.00965])
    assert np.isclose(fit_covar, expected_covar).all()


@pytest.mark.parametrize(
    "return_period, expected_results",
    [
        (
            100,
            {
                # Coles (2001) p. 60 example 3.4.1, RP-100
                'level': np.array([4.69]),
                'upper': np.array([5.]),
                'lower': np.array([4.38])
            }
        ),
        (
            10,
            {
                # Coles (2001) p. 60 example 3.4.1, RP-10
                'level': np.array([4.3]),
                'upper': np. array([4.4]),
                'lower': np.array([4.19])
            }
        )
    ]
)
def test_coles_return_levels(return_period, expected_results, fitted_model):
    """Return level test from Coles (2001) p. 60 example 3.4.1."""

    fit_values = fitted_model.return_level(return_period)

    # Rounding.
    fit_values = {key: np.round(val, 2) for (key, val) in fit_values.items()}

    checks = [
        np.isclose(fit_values[key], expected_results[key])
        for key in fit_values.keys()
    ]

    assert all(checks)
