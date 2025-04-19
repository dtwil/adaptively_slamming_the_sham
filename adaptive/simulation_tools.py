from itertools import product
from collections.abc import Iterable
from typing import Callable, Optional
import copy
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats

from tqdm.auto import tqdm
from cmdstanpy import CmdStanModel, CmdStanMCMC


@dataclass
class ExperimentResult:
    data: pd.DataFrame
    params: dict


class StanFitter:
    def __init__(self, model: CmdStanModel):
        self.model = model

    def fit(self, expt: ExperimentResult, **kwargs) -> CmdStanMCMC:
        return self.model.sample(data=self.to_dict(expt), **kwargs)

    def to_dict(self, expt: ExperimentResult) -> dict:
        raise NotImplementedError


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
    sample_error: np.ndarray
    pop_error: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "estimate": self.estimate,
                "se": self.se,
                "conf_lower": self.conf_lower,
                "conf_upper": self.conf_upper,
                "is_signif": self.is_signif,
                "correct_sign": self.correct_sign,
                "sample_error": self.sample_error,
                "pop_error": self.pop_error,
            }
        )

    def pop_rmse(self) -> float:
        """Calculate the population RMSE."""
        return np.sqrt(np.mean(self.pop_error**2))

    def sample_rmse(self) -> float:
        """Calculate the sample RMSE."""
        return np.sqrt(np.mean(self.sample_error**2))


class ExperimentSimulator:
    """Encapsulates MLM hyperparameters and simulation logic."""

    def __init__(
        self,
        mu_b: float,
        mu_theta: float,
        sigma_b: float,
        sigma_theta: float,
        sigma1: float,
        sigma0: float,
    ):
        self.mu_b = mu_b
        self.mu_theta = mu_theta
        self.sigma_b = sigma_b
        self.sigma_theta = sigma_theta
        self.sigma1 = sigma1
        self.sigma0 = sigma0

    def simulate(self, n: np.ndarray, p: np.ndarray) -> ExperimentResult:
        """Simulate a batch of experiments."""
        # Validate that treatment proportions are between 0 and 1
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError(
                f"All entries of p must lie in [0, 1]. "
                f"Found p.min()={np.min(p)}, p.max()={np.max(p)}"
            )

        n_arr, p_arr = n, p
        n1 = np.floor(p_arr * n_arr).astype(int)
        n0 = n_arr - n1
        sigma_y1_bar = self.sigma1 / np.sqrt(n1)
        sigma_y0_bar = self.sigma0 / np.sqrt(n0)
        theta = np.random.normal(self.mu_theta, self.sigma_theta, len(n_arr))
        b = np.random.normal(self.mu_b, self.sigma_b, len(n_arr))

        y1_bar = np.random.normal(theta + b, sigma_y1_bar)
        y0_bar = np.random.normal(b, sigma_y0_bar)

        df = pd.DataFrame(
            {
                "y1_bar": y1_bar,
                "y0_bar": y0_bar,
                "sigma_y1_bar": sigma_y1_bar,
                "sigma_y0_bar": sigma_y0_bar,
                "n": n_arr,
                "p": p_arr,
                "theta": theta,
                "b": b,
            }
        )
        params = {
            "mu_b": self.mu_b,
            "mu_theta": self.mu_theta,
            "sigma_b": self.sigma_b,
            "sigma_theta": self.sigma_theta,
            "sigma1": self.sigma1,
            "sigma0": self.sigma0,
        }
        return ExperimentResult(data=df, params=params)

    def simulate_sequence(
        self,
        n: Iterable[int],
        p_callback: Callable[[Optional[pd.DataFrame], int], float],
    ) -> pd.DataFrame:
        """
        Simulate experiments one at a time.
        p_callback(prev_df, n) should return the next treatment proportion.
        """
        dfs = []
        for n in n:
            prev = pd.concat(dfs, ignore_index=True) if dfs else None
            p = p_callback(prev, n)
            dfs.append(self.simulate(np.array([n]), np.array([p])).data)
        return pd.concat(dfs, ignore_index=True)


CHICK_J = 38
CHICK_N = 64
CHICK_MU_THETA = 0.09769112704348468
CHICK_MU_B = 0.004112476586286136
CHICK_SIGMA_THETA = 0.056385519973983916
CHICK_SIGMA_B = 0.0015924804430524674
CHICK_SIGMA1 = (32**0.5) * 0.04
CHICK_SIGMA0 = (32**0.5) * 0.04
CHICK_SIGMA_B_GRID = np.arange(0, 0.11, 0.01)
CHICK_SIMULATOR = ExperimentSimulator(
    mu_b=CHICK_MU_B,
    mu_theta=CHICK_MU_THETA,
    sigma_b=CHICK_SIGMA_B,
    sigma_theta=CHICK_SIGMA_THETA,
    sigma1=CHICK_SIGMA1,
    sigma0=CHICK_SIGMA0,
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


def simulate_experiments(params):
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
    return sim.simulate(
        params["n"],
        params["p"],
    ).data


def _get_estimates(expt_df, estimator_fn, alpha=0.05, **kwargs):
    """
    expt_df: A pandas DataFrame with experiment data
            (see the output of simulate_experiments)
    estimator_fn: The function to use for estimating the parameters
    alpha: The significance level for hypothesis testing

    Returns: An EstimatorResult object with the following attributes:
        - estimate: The estimated average response for the treated group
        - se: The standard error of the estimate
        - conf_lower: The lower bound of the confidence interval
        - conf_upper: The upper bound of the confidence interval
        - is_signif: A boolean indicating whether the estimate is significant
        - correct_sign: A boolean indicating whether the estimate has the correct sign
        - sample_error: The error of the estimate compared to the true parameter value
        - pop_error: The error of the estimate compared to the population parameter value
    """
    stats = estimator_fn(expt_df, alpha=alpha, **kwargs)
    est = stats["estimate"]
    se = stats["se"]
    lower = stats["conf_lower"]
    upper = stats["conf_upper"]
    signif = ~((lower < 0) & (0 < upper))

    # oracle metrics
    correct = np.sign(expt_df["theta"].to_numpy()) == np.sign(est)
    samp_err = expt_df["theta"].to_numpy() - est
    pop_err = expt_df.attrs["mu_theta"] - est

    return EstimatorResult(
        estimate=est,
        se=se,
        conf_lower=lower,
        conf_upper=upper,
        is_signif=signif,
        correct_sign=correct,
        sample_error=samp_err,
        pop_error=pop_err,
    )


def get_exposed_only_estimates(expt_df, alpha=0.05):
    """
    expt_df: A pandas DataFrame with experiment data
            (see the output of simulate_experiments)
    alpha: The significance level for hypothesis testing

    Returns: An EstimatorResult object with the following attributes:
        - estimate: The estimated average response for the treated group
        - se: The standard error of the estimate
        - conf_lower: The lower bound of the confidence interval
        - conf_upper: The upper bound of the confidence interval
        - is_signif: A boolean indicating whether the estimate is significant
        - correct_sign: A boolean indicating whether the estimate has the correct sign
        - sample_error: The error of the estimate compared to the true parameter value
        - pop_error: The error of the estimate compared to the population parameter value
    """

    def exposed_only_fn(expt_df, alpha=0.05):
        estimate = expt_df["y_1"]
        se = expt_df["sigma1"]
        z_value = stats.norm.ppf(1 - alpha / 2)
        conf_lower = estimate - z_value * se
        conf_upper = estimate + z_value * se

        return {
            "estimate": estimate,
            "se": se,
            "conf_lower": conf_lower,
            "conf_upper": conf_upper,
        }

    return _get_estimates(expt_df, exposed_only_fn, alpha=alpha)


def get_difference_estimates(expt_df, alpha=0.05):
    """
    expt_df: A pandas DataFrame with experiment data
            (see the output of simulate_experiments)
    alpha: The significance level for hypothesis testing

    Returns: An EstimatorResult object with the following attributes:
        - estimate: The estimated average response for the treated group
        - se: The standard error of the estimate
        - conf_lower: The lower bound of the confidence interval
        - conf_upper: The upper bound of the confidence interval
        - is_signif: A boolean indicating whether the estimate is significant
        - correct_sign: A boolean indicating whether the estimate has the correct sign
        - sample_error: The error of the estimate compared to the true parameter value
        - pop_error: The error of the estimate compared to the population parameter value
    """

    def difference_fn(expt_df, alpha=0.05):
        estimate = expt_df["y_1"] - expt_df["avg_control_response"]
        se = np.sqrt(expt_df["sigma1"] ** 2 + expt_df["sigma0"] ** 2)
        z_value = stats.norm.ppf(1 - alpha / 2)
        conf_lower = estimate - z_value * se
        conf_upper = estimate + z_value * se

        return {
            "estimate": estimate,
            "se": se,
            "conf_lower": conf_lower,
            "conf_upper": conf_upper,
        }

    return _get_estimates(expt_df, difference_fn, alpha=alpha)


def get_bayes_estimates(expt_df, model, alpha=0.05):
    """
    expt_df: A pandas DataFrame with experiment data
            (see the output of simulate_experiments)
    model: A cmdstanpy model object (not fitted)
    alpha: The significance level for hypothesis testing

    Returns: An EstimatorResult object with the following attributes:
        - estimate: The estimated average response for the treated group
        - se: The standard error of the estimate
        - conf_lower: The lower bound of the confidence interval
        - conf_upper: The upper bound of the confidence interval
        - is_signif: A boolean indicating whether the estimate is significant
        - correct_sign: A boolean indicating whether the estimate has the correct sign
        - sample_error: The error of the estimate compared to the true parameter value
        - pop_error: The error of the estimate compared to the population parameter value
    """

    def bayes_fn(expt_df, alpha=0.05, model=model):
        fit = model.sample(data=expt_df_to_dict(expt_df), show_progress=False)
        estimate = np.mean(fit.stan_variable("theta"), axis=0)
        se = np.std(fit.stan_variable("theta"), axis=0)

        thetas = fit.stan_variable("theta")
        conf_lower = np.quantile(thetas, alpha / 2, axis=0)
        conf_upper = np.quantile(thetas, 1 - alpha / 2, axis=0)

        return {
            "estimate": estimate,
            "se": se,
            "conf_lower": conf_lower,
            "conf_upper": conf_upper,
        }

    return _get_estimates(expt_df, bayes_fn, alpha=alpha, model=model)


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
    true_theta = estimates_df["estimate"] + estimates_df["sample_error"]

    prop_signif = np.mean(estimates_df["is_signif"])
    sample_mse = np.mean(estimates_df["sample_error"] ** 2)
    pop_mse = np.mean(estimates_df["pop_error"] ** 2)
    rank_corr = estimates_df["estimate"].corr(true_theta, method="spearman")

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
        expt_df = simulate_experiments(params)
        estimator_dfs = {
            name: res.to_frame()
            for name, res in {
                "exposed_only": get_exposed_only_estimates(expt_df),
                "difference": get_difference_estimates(expt_df),
                "bayes": get_bayes_estimates(expt_df, model=model),
            }.items()
        }

        for estimator_name, df in estimator_dfs.items():
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
