from collections.abc import Iterable
from typing import Callable, Optional
import copy
from dataclasses import dataclass
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from scipy import stats

from tqdm.auto import tqdm
from cmdstanpy import CmdStanModel, CmdStanMCMC


@dataclass
class ExperimentResult:
    data: pd.DataFrame
    params: dict


class StanFitter(ABC):
    def __init__(self, model: CmdStanModel):
        self.model = model

    def fit(self, expt: ExperimentResult, **kwargs) -> CmdStanMCMC:
        return self.model.sample(data=self.to_dict(expt), **kwargs)

    @abstractmethod
    def to_dict(self, expt: ExperimentResult) -> dict: ...


class StanFitterBasic(StanFitter):
    def to_dict(self, expt: ExperimentResult) -> dict:
        return {
            "J": len(expt.data),
            "j": list(range(1, len(expt.data) + 1)),
            "y1_bar": list(expt.data["y1_bar"]),
            "y0_bar": list(expt.data["y0_bar"]),
            "sigma_y1_bar": list(expt.data["sigma_y1_bar"]),
            "sigma_y0_bar": list(expt.data["sigma_y0_bar"]),
            "n": list(expt.data["n"]),
            "p": list(expt.data["p"]),
        }


class StanFitterBasicWithHyperparams(StanFitter):
    def to_dict(self, expt: ExperimentResult) -> dict:
        return {
            "J": len(expt.data),
            "j": list(range(1, len(expt.data) + 1)),
            "y1_bar": list(expt.data["y1_bar"]),
            "y0_bar": list(expt.data["y0_bar"]),
            "sigma_y1_bar": list(expt.data["sigma_y1_bar"]),
            "sigma_y0_bar": list(expt.data["sigma_y0_bar"]),
            "n": list(expt.data["n"]),
            "p": list(expt.data["p"]),
            "mu_b": expt.params["mu_b"],
            "mu_theta": expt.params["mu_theta"],
            "sigma_b": expt.params["sigma_b"],
            "sigma_theta": expt.params["sigma_theta"],
        }


@dataclass
class EstimatorResult:
    estimate: np.ndarray
    se: np.ndarray
    conf_lower: np.ndarray
    conf_upper: np.ndarray
    is_signif: np.ndarray
    correct_sign: np.ndarray
    samp_err: np.ndarray
    pop_err: np.ndarray

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "estimate": self.estimate,
                "se": self.se,
                "conf_lower": self.conf_lower,
                "conf_upper": self.conf_upper,
                "is_signif": self.is_signif,
                "correct_sign": self.correct_sign,
                "samp_err": self.samp_err,
                "pop_err": self.pop_err,
            }
        )

    def summary(self) -> pd.DataFrame:
        """Return a summary of the estimator results."""
        return pd.DataFrame(
            {
                "prop_signif": [np.mean(self.is_signif)],
                "sample_rmse": [self.sample_rmse()],
                "pop_rmse": [self.pop_rmse()],
                "type_s_rate": [self.type_s_rate()],
                "rank_corr": [self.rank_corr()],
            }
        )

    def pop_rmse(self) -> float:
        """Calculate the population RMSE."""
        return np.sqrt(np.mean(self.pop_err**2))

    def sample_rmse(self) -> float:
        """Calculate the sample RMSE."""
        return np.sqrt(np.mean(self.samp_err**2))

    def type_s_rate(self) -> float:
        """Calculate the type S error rate."""
        is_type_s_error = self.is_signif & ~self.correct_sign
        if len(is_type_s_error) > 0:
            return np.sum(is_type_s_error) / len(is_type_s_error)
        else:
            return 0.0

    def rank_corr(self) -> float:
        """Calculate the rank correlation between the estimate and the true theta."""
        theta = self.estimate + self.samp_err
        return stats.spearmanr(self.estimate, theta).statistic


class ExperimentSimulator(ABC):
    """Encapsulates MLM hyperparameters and simulation logic."""

    def __init__(
        self,
        mu_b: float,
        mu_theta: float,
        sigma_b: float,
        sigma_theta: float,
        sigma1: float,
        sigma0: float,
        identifier: dict,
    ):
        self.mu_b = mu_b
        self.mu_theta = mu_theta
        self.sigma_b = sigma_b
        self.sigma_theta = sigma_theta
        self.sigma1 = sigma1
        self.sigma0 = sigma0
        self.params = {
            "mu_b": mu_b,
            "mu_theta": mu_theta,
            "sigma_b": sigma_b,
            "sigma_theta": sigma_theta,
            "sigma1": sigma1,
            "sigma0": sigma0,
        }
        self.identifier = identifier

    @abstractmethod
    def simulate(self) -> ExperimentResult: ...


class StaticExperimentSimulator(ExperimentSimulator):
    def __init__(
        self,
        mu_b: float,
        mu_theta: float,
        sigma_b: float,
        sigma_theta: float,
        sigma1: float,
        sigma0: float,
        n: np.ndarray,
        p: np.ndarray,
        identifier: dict,
    ):
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError(
                f"All entries of p must lie in [0, 1]. "
                f"Found p.min()={np.min(p)}, p.max()={np.max(p)}"
            )
        super().__init__(
            mu_b, mu_theta, sigma_b, sigma_theta, sigma1, sigma0, identifier
        )
        self.n = n
        self.p = p

    def simulate(self) -> ExperimentResult:
        """Simulate a batch of experiments."""
        n1 = np.floor(self.p * self.n).astype(int)
        n0 = self.n - n1
        sigma_y1_bar = self.sigma1 / np.sqrt(n1)
        sigma_y0_bar = self.sigma0 / np.sqrt(n0)
        theta = np.random.normal(self.mu_theta, self.sigma_theta, len(self.n))
        b = np.random.normal(self.mu_b, self.sigma_b, len(self.n))

        y1_bar = np.random.normal(theta + b, sigma_y1_bar)
        y0_bar = np.random.normal(b, sigma_y0_bar)

        df = pd.DataFrame(
            {
                "y1_bar": y1_bar,
                "y0_bar": y0_bar,
                "sigma_y1_bar": sigma_y1_bar,
                "sigma_y0_bar": sigma_y0_bar,
                "n": self.n,
                "p": self.p,
                "theta": theta,
                "b": b,
            }
        )
        return ExperimentResult(data=df, params=self.params)


class AdaptiveExperimentSimulator(ExperimentSimulator):
    def __init__(
        self,
        mu_b: float,
        mu_theta: float,
        sigma_b: float,
        sigma_theta: float,
        sigma1: float,
        sigma0: float,
        J: int,
        p_callback: Callable[[ExperimentResult | None], float],
        n_callback: Callable[[ExperimentResult | None], int],
        identifier: dict,
    ):
        """
        Simulate experiments one at a time.
        p_callback(prev_df, n) should return the next treatment proportion.
        J is the total number of experiments to generate.
        """
        super().__init__(
            mu_b, mu_theta, sigma_b, sigma_theta, sigma1, sigma0, identifier
        )
        self.J = J
        self.p_callback = p_callback
        self.n_callback = n_callback

    def simulate(self) -> pd.DataFrame:
        # Simulate the first experiment
        static = StaticExperimentSimulator(**self.params)
        expt = static.simulate([self.n_callback(None)], [self.p_callback(None)])

        # simulate the rest of the experiments
        for j in range(1, self.J):
            next_expt = static.simulate(
                [self.n_callback(expt)], [self.p_callback(expt)]
            )
            merged_dfs = pd.concat([expt.data, next_expt.data], ignore_index=True)
            expt = ExperimentResult(data=merged_dfs, params=self.params)

        return expt


class Estimator(ABC):
    @abstractmethod
    def _compute(self, df: pd.DataFrame) -> dict: ...

    def result(
        self, expt: ExperimentResult, alpha: float = 0.05, **kwargs
    ) -> EstimatorResult:
        stats = self._compute(expt, alpha, **kwargs)
        est, se = stats["estimate"], stats["se"]
        lower, upper = stats["conf_lower"], stats["conf_upper"]
        signif = ~((lower < 0) & (0 < upper))
        correct = np.sign(expt.data["theta"].to_numpy()) == np.sign(est)
        samp_err = expt.data["theta"].to_numpy() - est
        pop_err = expt.params["mu_theta"] - est
        return EstimatorResult(
            est, se, lower, upper, signif, correct, samp_err, pop_err
        )


class ExposedOnlyEstimator(Estimator):
    def __init__(self):
        self.name = "Exposed Only"

    def _compute(self, expt: ExperimentResult, alpha: float, **kwargs) -> dict:
        df = expt.data
        z = stats.norm.ppf(1 - alpha / 2)
        est, se = df["y1_bar"], df["sigma_y1_bar"]
        return {
            "estimate": est,
            "se": se,
            "conf_lower": est - z * se,
            "conf_upper": est + z * se,
        }


class DifferenceEstimator(Estimator):
    def __init__(self):
        self.name = "Difference"

    def _compute(self, expt: ExperimentResult, alpha: float, **kwargs) -> dict:
        df = expt.data
        z = stats.norm.ppf(1 - alpha / 2)
        est = df["y1_bar"] - df["y0_bar"]
        se = np.sqrt(df["sigma_y1_bar"] ** 2 + df["sigma_y0_bar"] ** 2)
        return {
            "estimate": est,
            "se": se,
            "conf_lower": est - z * se,
            "conf_upper": est + z * se,
        }


class PosteriorMeanEstimator(Estimator):
    def __init__(self, fitter: StanFitter):
        super().__init__()
        self.fitter = fitter
        self.name = "Posterior Mean"

    def _compute(self, expt: ExperimentResult, alpha: float, **kwargs):
        fit = self.fitter.fit(expt=expt, **kwargs)
        thetas = fit.stan_variable("theta")
        est, se = np.mean(thetas, axis=0), np.std(thetas, axis=0)
        return {
            "estimate": est,
            "se": se,
            "conf_lower": np.quantile(thetas, alpha / 2, axis=0),
            "conf_upper": np.quantile(thetas, 1 - alpha / 2, axis=0),
        }


@dataclass
class SimulationAggregatorResult:
    """
    Encapsulates the logic for aggregating simulation results.
    """

    df: pd.DataFrame
    identifiers: list[str]

    def means(self) -> pd.DataFrame:
        """
        Returns the mean of the simulation results.
        """
        return self.df.groupby(self.identifiers).mean().reset_index()


class SimulationAggregator:
    """
    Encapsulates the logic for aggregating simulation results.
    Assumes all simulators have the same identifier keys.
    """

    def __init__(self, simulators: Iterable[ExperimentSimulator]):
        self.simulators = list(simulators)
        if self.simulators:
            # collect the key‐set of the first simulator
            common_keys = set(self.simulators[0].identifier.keys())
            # ensure all simulators share the same identifier keys
            for sim in self.simulators[1:]:
                if set(sim.identifier.keys()) != common_keys:
                    raise ValueError(
                        f"All simulators must have the same identifier keys. "
                        f"Expected {common_keys}, got {set(sim.identifier.keys())} from {sim}"
                    )

        self.identifiers = list(
            self.simulators[0].identifier.keys() if self.simulators else []
        )

    def simulate(
        self, estimators: Iterable[Estimator], num_reps: int, alpha: float, **kwargs
    ) -> SimulationAggregatorResult:
        df = pd.DataFrame()
        for simulator in tqdm(self.simulators, desc="Simulators", leave=False):
            for estimator in tqdm(estimators, desc="Estimators", leave=False):
                new_df = None
                for _ in tqdm(range(num_reps), desc="Repetition", leave=False):
                    expt = simulator.simulate()
                    summary_df = estimator.result(expt, alpha, **kwargs).summary()

                    if new_df is None:
                        new_df = summary_df
                    else:
                        new_df = pd.concat([new_df, summary_df], ignore_index=True)

                # Add the simulator and estimators to the DataFrame
                for id in simulator.identifier:
                    new_df[id] = simulator.identifier[id]
                new_df["estimator"] = estimator.name

                # Add the new DataFrame to the main DataFrame
                df = pd.concat([df, new_df], ignore_index=True)

        return SimulationAggregatorResult(df, list(self.identifiers) + ["estimator"])


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
