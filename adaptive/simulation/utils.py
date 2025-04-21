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

0


def next_p_star(expt: ExperimentResult | None, n: int, oracle: bool = False) -> float:
    """
    Calculate the p* value for the given experiment.
    n is the sample size of the next experiment.
    """
    if expt is None:
        return 0.5
    sigma0 = expt.params["sigma0"]
    sigma1 = expt.params["sigma1"]

    if oracle:
        sigma_b = expt.params["sigma_b"]
    else:
        sigma_b = sigma_b_hat(expt)

    numer = 1 + sigma0**2 / (n * sigma_b**2)
    denom = 1 + sigma0 / sigma1
    p_star = numer / denom

    return min(0.95, p_star)


def sigma_b_hat(expt: ExperimentResult) -> float:
    y1 = expt.data["y1_bar"]
    y0 = expt.data["y0_bar"]
    J = len(expt.data)

    var_b_hat = np.sum(((y1 - np.mean(y1)) * (y0 - np.mean(y0)))) / J
    if var_b_hat <= 0:
        return 1e-4

    return var_b_hat**0.5
