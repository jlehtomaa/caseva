from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from caseva.models import BlockMaximaModel


DIFF_ATOL = 0.01  # Allowed absolute tolerance for testing fitted parameters.
PATH_DATA = Path(__file__).parent.parent / "data"


@pytest.fixture
def fitted_model_sealevel():
    """Block maxima model fit to the Port Pirie sea level data (Coles 2001)."""

    data = pd.read_csv(PATH_DATA / "portpirie.csv")
    extremes = data["SeaLevel"]

    model = BlockMaximaModel(data=extremes)
    model.fit()

    return model


@pytest.fixture
def fitted_model_glassfiber():
    """Block maxima model fit to the (negated) glassfiber data (Coles 2001)."""

    data = pd.read_csv(PATH_DATA / "glass.csv")
    extremes = -data["Strength"]

    model = BlockMaximaModel(data=extremes)
    model.fit()

    return model


def test_coles_example_3_4_1_theta(fitted_model_sealevel):
    """
    Coles (2001) p. 59 example 3.4.1: Model parameters test.
    """

    fit_theta = fitted_model_sealevel.theta
    expected_theta = np.array([3.87, 0.198, -0.050])
    np.testing.assert_allclose(fit_theta, expected_theta, atol=DIFF_ATOL)


def test_coles_example_3_4_1_covar(fitted_model_sealevel):
    """
    Coles (2001) p. 59 example 3.4.1: Model parameter covariance test.
    """

    fit_covar = fitted_model_sealevel.covar
    expected_covar = np.array([
        [0.000780, 0.000197, -0.00107],
        [0.000197, 0.000410, -0.000778],
        [-0.00107, -0.000778, 0.00965]
    ])
    np.testing.assert_allclose(fit_covar, expected_covar, atol=DIFF_ATOL)


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
                'upper': np. array([4.41]),
                'lower': np.array([4.19])
            }
        )
    ]
)
def test_coles_example_3_4_1_return_levels(
    return_period,
    expected_results,
    fitted_model_sealevel
):
    """Return level test from Coles (2001) p. 60 example 3.4.1."""

    fit_values = fitted_model_sealevel.return_level(return_period)

    checks = [
        np.isclose(fit_values[key], expected_results[key], atol=DIFF_ATOL)
        for key in fit_values.keys()
    ]

    assert all(checks)


def test_coles_example_3_4_2_theta(fitted_model_glassfiber):
    """
    Coles (2001) p. 65 example 3.4.2: Model parameters test.
    """

    fit_theta = np.round(fitted_model_glassfiber.theta, 3)
    expected_theta = np.array([-1.64, 0.27, -0.084])
    np.testing.assert_allclose(fit_theta, expected_theta, atol=DIFF_ATOL)


def test_coles_example_3_4_2_covar(fitted_model_glassfiber):
    """
    Coles (2001) p. 65 example 3.4.2: Model parameter covariance test.
    """

    fit_covar = fitted_model_glassfiber.covar
    expected_covar = np.array([
        [0.00141, -0.000214, 0.000795],
        [-0.000214, 0.000652, -0.000441],
        [0.000795, -0.0000441, 0.00489]
    ])
    np.testing.assert_allclose(fit_covar, expected_covar, atol=DIFF_ATOL)
