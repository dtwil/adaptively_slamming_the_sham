import numpy as np
import os
import pickle
import re
from datetime import datetime

from .results import ExperimentResult
from .simulators import ExperimentSimulator
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


def theta_posterior_variance(simulator: ExperimentSimulator) -> float:
    """
    Returns a vector of the posterior variance of theta for each experiment.
    """
    params = simulator.params()
    sigma1 = params["sigma1"]
    sigma0 = params["sigma0"]
    p = simulator.p
    n = simulator.n
    sigma_theta = params["sigma_theta"]
    sigma_b = params["sigma_b"]

    V1 = sigma1**2 / (p * n)
    V0 = sigma0**2 / ((1 - p) * n)

    A = 1.0 / V1 + 1.0 / (sigma_theta**2)
    B = 1.0 / V1 + 1.0 / V0 + 1.0 / (sigma_b**2)
    C = 1.0 / V1

    denom = A * B - C**2
    theta_post_var = B / denom

    return theta_post_var


def load_latest_simulation(data_dir, prefix):
    # Regex to match filenames like: sim_results_v2_20250429_143512.pkl
    pattern = re.compile(rf"{prefix}_(\d{{8}}_\d{{6}})\.pkl")

    # List and filter files
    files = [f for f in os.listdir(data_dir) if pattern.match(f)]

    if not files:
        raise FileNotFoundError("No compatible simulation files found.")

    # Sort by timestamp extracted from filename
    files.sort(key=lambda f: pattern.match(f).group(1), reverse=True)
    latest_file = os.path.join(data_dir, files[0])

    # Load and verify schema
    with open(latest_file, "rb") as f:
        data = pickle.load(f)

    return data


def save_simulation(results, data_dir, prefix):
    """
    Saves a simulation result with a timestamped filename.

    Parameters:
        results (any): The data to save.
        metadata (dict): Optional metadata (e.g. schema_version, config, timestamp).
        data_dir (str): Directory to save the file in.
        prefix (str): Filename prefix before the timestamp.
    """
    os.makedirs(data_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.pkl"
    full_path = os.path.join(data_dir, filename)

    with open(full_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved simulation to {full_path}")
    return full_path
