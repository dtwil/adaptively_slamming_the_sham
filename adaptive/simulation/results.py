from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class ExperimentResult:
    data: pd.DataFrame
    params: dict


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
