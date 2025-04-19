from itertools import product
from collections.abc import Iterable
import copy

import pandas as pd
import numpy as np
from scipy import stats

from tqdm.auto import tqdm


CHICK_NUM_EXPTS = 38
CHICK_NUM_SUBJECTS_PER_EXPT = 64
CHICK_MU_THETA = 0.09769112704348468
CHICK_MU_B = 0.004112476586286136
CHICK_SIGMA_THETA = 0.056385519973983916
CHICK_SIGMA_B = 0.0015924804430524674
CHICK_SIGMA_TREATMENT = (32**0.5) * 0.04
CHICK_SIGMA_CONTROL = (32**0.5) * 0.04
CHICK_SIGMA_B_GRID = np.arange(0, 0.11, 0.01)


def expt_df_to_dict(expt_df, remove_hyperparams=True):
    to_dict = expt_df.to_dict(orient="list")
    to_dict["num_expts"] = len(expt_df)
    to_dict["expt_id"] = list(range(1, len(expt_df) + 1))

    if remove_hyperparams:
        if "theta" in to_dict:
            del to_dict["theta"]
        if "b" in to_dict:
            del to_dict["b"]

    return to_dict


def expt_df_to_params(expt_df):
    expt_df_copy = copy.deepcopy(expt_df)
    params = expt_df_copy.attrs
    params["num_subjects_per_expt"] = expt_df_copy["num_subjects_per_expt"]
    params["prop_treatment"] = expt_df_copy["prop_treatment"]
    return params


def get_chick_data(chick_data_path):
    chicks = pd.read_table(chick_data_path, sep="\\s+")
    chicks["exposed_est"] -= 1
    chicks["sham_est"] -= 1
    chick_data = {
        "num_expts": len(chicks),
        "avg_treated_response": chicks["exposed_est"],
        "avg_control_response": chicks["sham_est"],
        "treated_se": chicks["exposed_se"],
        "control_se": chicks["sham_se"],
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
    Simulate a set of experiments according to the multilevel model:

    b_j ~ N(mu_b, sigma_b^2)
    theta_j ~ N(mu_theta, sigma_theta^2)

    y_{j0} ~ N(b_j, sigma_control^2 / n_{j0})
    y_{j1} ~ N(b_j + theta_j, sigma_treatment^2 / n_{j1})

    where j is the experiment index, n_{j0} is the number of control subjects
    in experiment j, and n_{j1} is the number of treated subjects in experiment j.
    Note that n_{j1} = prop_treatment[j] * num_subjects_per_expt[j].

    Returns: a dictionary whose entries are the true parameters of the model,
    the number of experiments, the average treated and control responses,
    the standard errors of the treated and control responses, and the
    experiment IDs.
    """
    num_subjects_per_expt = params["num_subjects_per_expt"]
    prop_treatment = params["prop_treatment"]
    mu_b = params["mu_b"]
    mu_theta = params["mu_theta"]
    sigma_b = params["sigma_b"]
    sigma_theta = params["sigma_theta"]
    sigma_treatment = params["sigma_treatment"]
    sigma_control = params["sigma_control"]

    assert isinstance(num_subjects_per_expt, np.ndarray)
    assert isinstance(prop_treatment, np.ndarray)
    assert len(num_subjects_per_expt) == len(prop_treatment)
    assert sigma_theta >= 0 and sigma_b >= 0
    assert sigma_treatment >= 0 and sigma_control >= 0

    num_expts = len(num_subjects_per_expt)
    num_treated = np.floor(prop_treatment * num_subjects_per_expt).astype(int)
    num_control = num_subjects_per_expt - num_treated

    sigma_y1 = sigma_treatment / np.sqrt(num_treated)
    sigma_y0 = sigma_control / np.sqrt(num_control)

    theta = np.random.normal(mu_theta, sigma_theta, num_expts)
    b = np.random.normal(mu_b, sigma_b, num_expts)

    expt_df = pd.DataFrame(
        {
            "avg_treated_response": np.random.normal(theta + b, sigma_y1),
            "avg_control_response": np.random.normal(b, sigma_y0),
            "treated_se": sigma_y1,
            "control_se": sigma_y0,
            "num_subjects_per_expt": num_subjects_per_expt,
            "prop_treatment": prop_treatment,
            "theta": theta,
            "b": b,
        }
    )
    expt_df.attrs = {
        "mu_b": mu_b,
        "mu_theta": mu_theta,
        "sigma_b": sigma_b,
        "sigma_theta": sigma_theta,
        "sigma_treatment": sigma_treatment,
        "sigma_control": sigma_control,
    }

    return expt_df


def _get_estimates(expt_df, estimator_fn, alpha=0.05, **kwargs):
    """
    expt_df: A pandas DataFrame with experiment data
            (see the output of simulate_experiments)
    estimator_fn: The function to use for estimating the parameters
    alpha: The significance level for hypothesis testing

    Returns: A pandas DataFrame with num_expt rows and the following columns:
        - estimate: The estimated average response for the treated group
        - se: The standard error of the estimate
        - conf_lower: The lower bound of the confidence interval
        - conf_upper: The upper bound of the confidence interval
        - is_signif: A boolean indicating whether the estimate is significant
        - correct_sign: A boolean indicating whether the estimate has the correct sign
        - error: The error of the estimate compared to the true parameter value
    """
    stats = estimator_fn(expt_df, alpha=alpha, **kwargs)
    estimate = stats["estimate"]
    se = stats["se"]
    conf_lower = stats["conf_lower"]
    conf_upper = stats["conf_upper"]
    significant = ~((conf_lower < 0) & (0 < conf_upper))

    eval_df = pd.DataFrame(
        {
            "estimate": estimate,
            "se": se,
            "conf_lower": conf_lower,
            "conf_upper": conf_upper,
            "is_signif": significant,
        }
    )

    # Add oracle metrics (only knowable in simulations)
    eval_df["correct_sign"] = np.sign(expt_df["theta"]) == np.sign(estimate)
    eval_df["sample_error"] = expt_df["theta"] - estimate
    eval_df["pop_error"] = expt_df.attrs["mu_theta"] - estimate

    return eval_df


def get_exposed_only_estimates(expt_df, alpha=0.05):
    """
    expt_df: A pandas DataFrame with experiment data
            (see the output of simulate_experiments)
    alpha: The significance level for hypothesis testing

    Returns: A pandas DataFrame with num_expt rows and the following columns:
        - estimate: The estimated average response for the treated group
        - se: The standard error of the estimate
        - conf_lower: The lower bound of the confidence interval
        - conf_upper: The upper bound of the confidence interval
        - is_signif: A boolean indicating whether the estimate is significant
        - correct_sign: A boolean indicating whether the estimate has the correct sign
        - error: The error of the estimate compared to the true parameter value
    """

    def exposed_only_fn(expt_df, alpha=0.05):
        estimate = expt_df["avg_treated_response"]
        se = expt_df["treated_se"]
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

    Returns: A pandas DataFrame with num_expt rows and the following columns:
        - estimate: The estimated average response for the treated group
        - se: The standard error of the estimate
        - conf_lower: The lower bound of the confidence interval
        - conf_upper: The upper bound of the confidence interval
        - is_signif: A boolean indicating whether the estimate is significant
        - correct_sign: A boolean indicating whether the estimate has the correct sign
        - error: The error of the estimate compared to the true parameter value
    """

    def difference_fn(expt_df, alpha=0.05):
        estimate = expt_df["avg_treated_response"] - expt_df["avg_control_response"]
        se = np.sqrt(expt_df["treated_se"] ** 2 + expt_df["control_se"] ** 2)
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

    Returns: A pandas DataFrame with num_expt rows and the following columns:
        - estimate: The estimated average response for the treated group
        - se: The standard error of the estimate
        - conf_lower: The lower bound of the confidence interval
        - conf_upper: The upper bound of the confidence interval
        - is_signif: A boolean indicating whether the estimate is significant
        - correct_sign: A boolean indicating whether the estimate has the correct sign
        - error: The error of the estimate compared to the true parameter value
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
            "exposed_only": get_exposed_only_estimates(expt_df),
            "difference": get_difference_estimates(expt_df),
            "bayes": get_bayes_estimates(expt_df, model=model),
        }

        for estimator_name, estimates_df in estimator_dfs.items():
            evaluation = evaluate_estimates(estimates_df)
            evaluation["params"] = [params]  # Ensure the column is of type object
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

    # need to convert the prop_treatment column to a tuple for grouping
    if "prop_treatment" in params_with_mult_vals:
        eval["prop_treatment"] = eval.apply(
            lambda x: tuple(x["params"]["prop_treatment"]), axis=1
        )

    eval.drop(columns=["params"], inplace=True)
    grouped = eval.groupby(["estimator"] + params_with_mult_vals).mean().reset_index()

    # convert prop_treatment back to a numpy array
    if "prop_treatment" in params_with_mult_vals:
        grouped["prop_treatment"] = grouped.apply(
            lambda x: np.array(x["prop_treatment"]), axis=1
        )

    return grouped
