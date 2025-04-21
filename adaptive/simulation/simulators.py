from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Callable

from .results import ExperimentResult


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
        n_initial = self.n_callback(None)
        static = StaticExperimentSimulator(
            **self.params,
            n=np.array([n_initial]),
            p=np.array([self.p_callback(None, n_initial)]),
            identifier=self.identifier,
        )
        expt = static.simulate()

        # simulate the rest of the experiments
        for j in range(1, self.J):
            next_n = self.n_callback(expt)
            static = StaticExperimentSimulator(
                **self.params,
                n=np.array([next_n]),
                p=np.array([self.p_callback(None, next_n)]),
                identifier=self.identifier,
            )
            next_expt = static.simulate()
            merged_dfs = pd.concat([expt.data, next_expt.data], ignore_index=True)
            expt = ExperimentResult(data=merged_dfs, params=self.params)

        return expt
