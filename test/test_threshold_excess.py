from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from caseva.models import ThresholdExcessModel


DIFF_ATOL = 0.01  # Allowed absolute tolerance for testing fitted parameters.
PATH_DATA = Path(__file__).parent.parent / "data"


@pytest.fixture
def fitted_model_rainfall():
    """Threshold excess model fit to the daily rainflall data (Coles 2001)."""
    data = pd.read_csv(PATH_DATA / "rain.csv", parse_dates=[0])
    years = {date.year for date in data["Date"]}
    num_years = max(years) - min(years) + 1
    threshold = 30

    model = ThresholdExcessModel()

    model.fit(data=data["Rainfall"], threshold=threshold, num_years=num_years)
    return model


@pytest.fixture
def fitted_model_dowjones():
    """Threshold excess model fit to the daily rainflall data (Coles 2001)."""
    data = pd.read_csv(PATH_DATA / "dowjones.csv", parse_dates=["Date"])
    years = {date.year for date in data["Date"]}
    num_years = max(years) - min(years) + 1
    threshold = 2

    transformed_data = (
        np.log(data["Index"])[1:].values
        - np.log(data["Index"])[:-1].values
    ) * 100.

    model = ThresholdExcessModel()

    model.fit(data=transformed_data, threshold=threshold, num_years=num_years)
    return model


def test_coles_example_4_4_1_theta(fitted_model_rainfall):
    """
    Coles (2001) p. 84 example 4.4.1: Model parameters test.
    """
    fit_theta = fitted_model_rainfall.theta
    expected_theta = np.array([7.44, 0.184])

    np.testing.assert_allclose(fit_theta, expected_theta, atol=DIFF_ATOL)


def test_coles_example_4_4_1_covar(fitted_model_rainfall):
    """
    Coles (2001) p. 84 example 4.4.1: Model parameter covariance test.
    """

    fit_covar = fitted_model_rainfall.augmented_covar
    expected_covar = np.array([
        [4.9e-7, 0.0,     0.0],
        [0.0,    0.9188, -0.0655],
        [0.0,   -0.0655,  0.0102]
    ])

    np.testing.assert_allclose(fit_covar, expected_covar, atol=DIFF_ATOL)


def test_coles_example_4_4_1_return_level_100(fitted_model_rainfall):
    """
    Coles (2001) p. 86 example 4.4.1: Return period 100 test.
    """

    fit_values = fitted_model_rainfall.return_level(100)

    expected_values = {
        'level': np.array([106.34]),
        'upper': np.array([147.06]),
        'lower': np.array([65.62])
        }

    checks = [
        np.isclose(fit_values[key], expected_values[key], atol=DIFF_ATOL)
        for key in fit_values.keys()
    ]

    assert all(checks), fit_values


def test_coles_example_4_4_2_theta(fitted_model_dowjones):
    """
    Coles (2001) p. 84 example 4.4.1: Model parameters test.
    """
    fit_theta = fitted_model_dowjones.theta
    expected_theta = np.array([0.495, 0.288])

    np.testing.assert_allclose(fit_theta, expected_theta, atol=DIFF_ATOL)


def test_coles_example_4_4_2_exeedance_freq(fitted_model_dowjones):

    fit_proba = fitted_model_dowjones.thresh_exc_proba
    expected_proba = 0.028

    np.testing.assert_allclose(fit_proba, expected_proba, atol=0.001)
