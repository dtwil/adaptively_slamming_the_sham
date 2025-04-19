from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy import stats

from .results import ExperimentResult, EstimatorResult
from .fitters import StanFitter


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
