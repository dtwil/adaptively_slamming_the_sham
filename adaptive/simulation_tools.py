from itertools import product
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
        name: str | None = None,
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
        self.name = name

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
        name: str | None = None,
    ):
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError(
                f"All entries of p must lie in [0, 1]. "
                f"Found p.min()={np.min(p)}, p.max()={np.max(p)}"
            )
        super().__init__(mu_b, mu_theta, sigma_b, sigma_theta, sigma1, sigma0, name)
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
        name: str | None = None,
    ):
        """
        Simulate experiments one at a time.
        p_callback(prev_df, n) should return the next treatment proportion.
        J is the total number of experiments to generate.
        """
        super().__init__(mu_b, mu_theta, sigma_b, sigma_theta, sigma1, sigma0, name)
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


def simulate_and_estimate(
    simulators: Iterable[ExperimentSimulator],
    estimators: Iterable[Estimator],
    num_reps: int,
    alpha: float = 0.05,
    **kwargs: Optional[dict],
) -> pd.DataFrame:
    df = pd.DataFrame()
    for simulator in tqdm(simulators, desc="Simulators", leave=False):
        for estimator in tqdm(estimators, desc="Estimators", leave=False):
            new_df = None
            for i in tqdm(range(num_reps), desc="Repetition", leave=False):
                expt = simulator.simulate()
                summary_df = estimator.result(expt, alpha, **kwargs).summary()
                summary_df["iteration"] = i + 1

                if new_df is None:
                    new_df = summary_df
                else:
                    new_df = pd.concat([new_df, summary_df], ignore_index=True)

            # Add the simulator and estimators to the DataFrame
            new_df["simulator"] = simulator.name
            new_df["estimator"] = estimator.name

            # Add the new DataFrame to the main DataFrame
            df = pd.concat([df, new_df], ignore_index=True)
    return df


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
    name="chick_simulator",
)


def expt_df_to_dict(expt_df, remove_hyperparams=True):
    to_dict = expt_df.to_dict(orient="list")
    to_dict["J"] = len(expt_df)
    to_dict["j"] = list(range(1, len(expt_df) + 1))

    if remove_hyperparams:
        if "theta" in to_dict:
            del to_dict["theta"]
        if "b" in to_dict:
            del to_dict["b"]

    return to_dict


def expt_df_to_params(expt_df):
    expt_df_copy = copy.deepcopy(expt_df)
    params = expt_df_copy.attrs
    params["n"] = expt_df_copy["n"]
    params["p"] = expt_df_copy["p"]
    return params


def get_chick_data(chick_data_path):
    chicks = pd.read_table(chick_data_path, sep="\\s+")
    chicks["exposed_est"] -= 1
    chicks["sham_est"] -= 1
    chick_data = {
        "num_expts": len(chicks),
        "y_1": chicks["exposed_est"],
        "avg_control_response": chicks["sham_est"],
        "sigma1": chicks["exposed_se"],
        "sigma0": chicks["sham_se"],
        "expt_id": list(range(1, len(chicks) + 1)),
    }
    return chick_data


def posterior_summary(model, df):
    fit = model.sample(data=expt_df_to_dict(df), show_progress=False)
    return {
        "mu_theta": np.mean(fit.theta),
        "mu_b": np.mean(fit.b),
        "sigma_theta": np.std(np.mean(fit.theta, axis=0)),
        "sigma_b": np.std(np.mean(fit.b, axis=0)),
    }


def simulate_experiments(params) -> ExperimentResult:
    """
    Wrapper maintaining old API:
      params must include keys
      ['n','p','mu_b','mu_theta',
       'sigma_b','sigma_theta','sigma1','sigma0']
    """
    sim = ExperimentSimulator(
        mu_b=params["mu_b"],
        mu_theta=params["mu_theta"],
        sigma_b=params["sigma_b"],
        sigma_theta=params["sigma_theta"],
        sigma1=params["sigma1"],
        sigma0=params["sigma0"],
    )
    return sim.simulate(params["n"], params["p"])


def evaluate_estimates(estimates_df):
    """
    estimates_df: A pandas DataFrame of the type returned by the above 'estimates' functions.

    Returns: A pandas DataFrame with the following columns:
        - prop_signif: The proportion of estimates that are significant
        - sample_mse: The mean of (estimate[j] - theta[j])^2
        - pop_mse: The mean of (estimate[j] - mu_theta)^2
        - type_s_rate: The type S error rate (the proportion of estimates that are significant but have the wrong sign)

        Note that this DataFrame has only one row.
    """
    true_theta = estimates_df["estimate"] + estimates_df["samp_err"]

    prop_signif = np.mean(estimates_df["is_signif"])
    sample_mse = np.mean(estimates_df["samp_err"] ** 2)
    pop_mse = np.mean(estimates_df["pop_err"] ** 2)
    rank_corr = estimates_df["estimate"].corr(true_theta, method="spearman").statistic

    is_signif = estimates_df["is_signif"]
    correct_sign = estimates_df["correct_sign"]
    is_type_s_error = is_signif & ~correct_sign
    type_s_rate = (
        len(estimates_df[is_type_s_error]) / len(estimates_df)
        if len(estimates_df) > 0
        else 0
    )

    return pd.DataFrame(
        {
            "prop_signif": [prop_signif],
            "sample_mse": [sample_mse],
            "pop_mse": [pop_mse],
            "type_s_rate": [type_s_rate],
            "rank_corr": [rank_corr],
        }
    )


def repeat_inferences(model, reps, params):
    evaluations = pd.DataFrame()

    for i in tqdm(range(reps), desc="Repetition", leave=False):
        expt = simulate_experiments(params)
        new_dfs = {
            name: est.estimate(expt).to_frame()
            for name, est in {
                "exposed_only": ExposedOnlyEstimator(alpha=0.05),
                "difference": DifferenceEstimator(alpha=0.05),
                "bayes": BayesEstimator(model=model, alpha=0.05),
            }.items()
        }

        for estimator_name, df in new_dfs.items():
            evaluation = evaluate_estimates(df)
            evaluation["params"] = [params]
            evaluation["estimator"] = estimator_name
            evaluation["iteration"] = i + 1
            evaluations = pd.concat([evaluations, evaluation], ignore_index=True)

    return evaluations


def evaluate_params(model, reps, params):
    # every value in params must be a list
    assert all(isinstance(value, Iterable) for value in params.values())

    evaluation = pd.DataFrame()

    # Generate all combinations of parameter values
    param_keys = list(params.keys())
    param_combinations = list(product(*params.values()))

    for i, param_values in tqdm(
        enumerate(param_combinations),
        desc="Parameter Combination",
        leave=False,
        total=len(param_combinations),
    ):
        # Create a dictionary for the current combination of parameters
        current_params = dict(zip(param_keys, param_values))
        eval_current_params = repeat_inferences(model, reps, current_params)
        evaluation = pd.concat([evaluation, eval_current_params], ignore_index=True)

    return evaluation


def evaluate_params_means(model, reps, params):
    params_with_mult_vals = [p for p in params if len(params[p]) > 1]

    eval = evaluate_params(model, reps, params)
    eval = eval.drop(columns=["iteration"])

    for variable in params_with_mult_vals:
        eval[variable] = eval.apply(lambda x: x["params"][variable], axis=1)

    # need to convert the p column to a tuple for grouping
    if "p" in params_with_mult_vals:
        eval["p"] = eval.apply(lambda x: tuple(x["params"]["p"]), axis=1)

    eval.drop(columns=["params"], inplace=True)
    grouped = eval.groupby(["estimator"] + params_with_mult_vals).mean().reset_index()

    # convert p back to a numpy array
    if "p" in params_with_mult_vals:
        grouped["p"] = grouped.apply(lambda x: np.array(x["p"]), axis=1)

    return grouped
