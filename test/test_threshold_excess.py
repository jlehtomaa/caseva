
import pytest
import pandas as pd
import numpy as np
from caseva.models import ThresholdExcessModel


@pytest.fixture
def fitted_model():
    """Threshold excess model fit to the daily rainflall data (Coles 2001)."""
    data = pd.read_csv("./data/rain.csv", parse_dates=[0])
    years = [date.year for date in data["Date"]]
    num_years = max(years) - min(years) + 1
    threshold = 30

    model = ThresholdExcessModel(
        data=data["Rainfall"], threshold=threshold, num_years=num_years)

    model.fit()
    return model


def test_coles_example_4_4_1_theta(fitted_model):
    """
    Coles (2001) p. 84 example 4.4.1: Model parameters test.
    """
    fit_theta = fitted_model.theta.round(3)
    true_theta = np.array([7.44 , 0.184])
    assert np.isclose(fit_theta, true_theta).all()


def test_coles_example_4_4_1_covar(fitted_model):
    """
    Coles (2001) p. 84 example 4.4.1: Model parameter covariance test.
    """

    fit_covar = np.diag(fitted_model.covar).round(4)
    true_covar = np.array([0.9188, 0.0102])
    assert np.isclose(fit_covar, true_covar).all()


def test_coles_example_4_4_1_return_level_100(fitted_model):
    """
    Coles (2001) p. 86 example 4.4.1: Return period 100 test.
    """

    fit_values = fitted_model.return_level(100)

    # Rounding.
    fit_values = {key: val.round(1) for (key, val) in fit_values.items()}

    true_values = {
        'level': np.array([106.3]),
        'upper': np.array([147.2]),
        'lower': np.array([65.5])
        }

    checks = [
        np.isclose(fit_values[key], true_values[key]) for key in fit_values.keys()]

    assert all(checks)


def test_exceedance_probability(fitted_model):
    """
    Coles (2001) p. 86 example 4.4.1: Exceedance probability parameter test.
    (The parameter 'zeta' in the book.)
    """

    true_value = 0.00867
    fit_value = round(len(fitted_model.extremes) / len(fitted_model.data), 5)
    assert np.isclose(true_value, fit_value)
