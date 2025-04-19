import numpy as np

from .results import ExperimentResult
from .simulators import StaticExperimentSimulator

CHICK_J = 38
CHICK_N = 64
CHICK_MU_THETA = 0.09769112704348468
CHICK_MU_B = 0.004112476586286136
CHICK_SIGMA_THETA = 0.056385519973983916
CHICK_SIGMA_B = 0.0015924804430524674
CHICK_SIGMA1 = (32**0.5) * 0.04
CHICK_SIGMA0 = (32**0.5) * 0.04
CHICK_SIGMA_B_GRID = np.arange(0, 0.11, 0.01)
CHICK_SIMULATOR = StaticExperimentSimulator(
    mu_b=CHICK_MU_B,
    mu_theta=CHICK_MU_THETA,
    sigma_b=CHICK_SIGMA_B,
    sigma_theta=CHICK_SIGMA_THETA,
    sigma1=CHICK_SIGMA1,
    sigma0=CHICK_SIGMA0,
    n=np.array([CHICK_N] * CHICK_J),
    p=np.array([0.5] * CHICK_J),
    identifier={"name": "chick_simulator"},
)


def next_p_star(expt: ExperimentResult, n: int) -> float:
    """
    Calculate the p* value for the given experiment.
    n is the sample size of the next experiment.
    """
    sigma0 = expt.params["sigma0"]
    sigma1 = expt.params["sigma1"]
    sigma_b = expt.params["sigma_b"]

    numer = 1 + sigma0**2 / (n * sigma_b**2)
    denom = 1 + sigma0 / sigma1
    p_star = numer / denom

    return p_star
