
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

    fit_theta = fitted_model.theta.round(3)
    true_theta = np.array([ 3.875, 0.198, -0.05])
    assert np.isclose(fit_theta, true_theta).all()


def test_coles_example_3_4_1_covar(fitted_model):
    """
    Coles (2001) p. 59 example 3.4.1: Model parameter covariance test.
    """

    fit_covar = np.diag(fitted_model.covar).round(5)
    true_covar = np.array([0.00078, 0.00041, 0.00965])
    assert np.isclose(fit_covar, true_covar).all()


def test_coles_example_3_4_1_return_level_10(fitted_model):
    """
    Coles (2001) p. 60 example 3.4.1: Return period 10 test.
    """

    fit_values = fitted_model.return_level(10)

    # Rounding.
    fit_values = {key: val.round(2) for (key, val) in fit_values.items()}

    true_values = {
        'level': np.array([4.3]),
        'upper': np. array([4.4]),
        'lower': np.array([4.19])
        }

    checks = [
        np.isclose(fit_values[key], true_values[key]) for key in fit_values.keys()]

    assert all(checks)


def test_coles_example_3_4_1_return_level_100(fitted_model):
    """
    Coles (2001) p. 60 example 3.4.1: Return period 100 test.
    """

    fit_values = fitted_model.return_level(100)

    # Rounding.
    fit_values = {key: val.round(2) for (key, val) in fit_values.items()}

    true_values = {
        'level': np.array([4.69]),
        'upper': np.array([5.]),
        'lower': np.array([4.38])
        }

    checks = [
        np.isclose(fit_values[key], true_values[key]) for key in fit_values.keys()]

    assert all(checks)
    