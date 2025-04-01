import pandas as pd
import numpy as np
from scipy import stats


CHICK_NUM_EXPTS = 38
CHICK_NUM_SUBJECTS_PER_EXPT = 64
CHICK_MU_THETA = 0.09769112704348468
CHICK_MU_B = 0.004112476586286136
CHICK_SIGMA_THETA = 0.056385519973983916
CHICK_SIGMA_B = 0.0015924804430524674
CHICK_SIGMA_TREATMENT = (32**0.5) * 0.04
CHICK_SIGMA_CONTROL = (32**0.5) * 0.04
CHICK_SIGMA_B_GRID = np.arange(0, 0.11, 0.01)


def simulate_experiments(
    num_subjects_per_expt,
    prop_treatment,
    mu_b,
    mu_theta,
    sigma_b,
    sigma_theta,
    sigma_treatment,
    sigma_control,
):
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

    # need to return a dict because cmdstanpy expects a dict
    # also need to keep true_params to evaluate MSE, type S error, etc
    return {
        "true_params": {
            "mu_b": mu_b,
            "mu_theta": mu_theta,
            "sigma_b": sigma_b,
            "sigma_theta": sigma_theta,
            "sigma_treatment": sigma_treatment,
            "sigma_control": sigma_control,
            "theta": theta,
            "b": b,
        },
        "num_expts": len(num_subjects_per_expt),
        "avg_treated_response": np.random.normal(theta + b, sigma_y1),
        "avg_control_response": np.random.normal(b, sigma_y0),
        "treated_se": sigma_y1,
        "control_se": sigma_y0,
        "expt_id": list(range(1, num_expts + 1)),
    }


def _estimates_helper(data_dict, estimate, se, conf_lower, conf_upper):
    # an observation is significant if 0 is not in the interval
    significant = ~((conf_lower < 0) & (0 < conf_upper))
    correct_sign = np.sign(data_dict["true_params"]["mu_theta"]) == np.sign(estimate)
    error = data_dict["true_params"]["theta"] - estimate

    return pd.DataFrame(
        {
            "estimate": estimate,
            "se": se,
            "conf_lower": conf_lower,
            "conf_upper": conf_upper,
            "is_signif": significant,
            "correct_sign": correct_sign,
            "error": error,
        }
    )


def estimates_exposed_only(data_dict, alpha=0.05):
    """
    data_dict: A data dictionary (see the output of simulate_experiments)
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
    z_value = stats.norm.ppf(1 - alpha / 2)
    estimate = data_dict["avg_treated_response"]
    se = data_dict["treated_se"]
    conf_lower = estimate - z_value * se
    conf_upper = estimate + z_value * se

    return _estimates_helper(data_dict, estimate, se, conf_lower, conf_upper)


def estimates_difference(data_dict, alpha=0.05):
    """
    data_dict: A data dictionary (see the output of simulate_experiments)
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
    z_value = stats.norm.ppf(1 - alpha / 2)
    estimate = data_dict["avg_treated_response"] - data_dict["avg_control_response"]
    se = np.sqrt(data_dict["treated_se"] ** 2 + data_dict["control_se"] ** 2)
    conf_lower = estimate - z_value * se
    conf_upper = estimate + z_value * se

    return _estimates_helper(data_dict, estimate, se, conf_lower, conf_upper)


def estimates_posterior(data_dict, model, alpha=0.05):
    """
    data_dict: A data dictionary (see the output of simulate_experiments)
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
    fit = model.sample(data=data_dict, adapt_delta=0.9, show_progress=False)
    estimate = np.mean(fit.stan_variable("theta"), axis=0)
    se = np.std(fit.stan_variable("theta"), axis=0)

    thetas = fit.stan_variable("theta")
    conf_lower = np.quantile(thetas, alpha / 2, axis=0)
    conf_upper = np.quantile(thetas, 1 - alpha / 2, axis=0)

    return _estimates_helper(data_dict, estimate, se, conf_lower, conf_upper)


def get_summary(df):
    statistics = {}
    statistics["prop_signif"] = np.mean(df["is_signif"])
    statistics["mse"] = np.mean(df["error"] ** 2)
    statistics["type_s_rate"] = (
        len(df[df["is_signif"] & ~df["correct_sign"]]) / len(df) if len(df) > 0 else 0
    )
    return statistics


def repeat_inferences(
    model,
    num_repetitions,
    num_subjects_per_expt,
    prop_treatment,
    mu_b,
    mu_theta,
    sigma_b,
    sigma_theta,
    sigma_treatment,
    sigma_control,
    show_progress=False,
):
    summaries = {
        method: {"prop_signif": [], "type_s_rate": [], "mse": []}
        for method in ["exposed_only", "difference", "posterior"]
    }

    for i in range(num_repetitions):
        fake_data = simulate_experiments(
            num_subjects_per_expt,
            prop_treatment,
            mu_b,
            mu_theta,
            sigma_b,
            sigma_theta,
            sigma_treatment,
            sigma_control,
        )

        exposed_only = estimates_exposed_only(fake_data)
        difference = estimates_difference(fake_data)
        posterior = estimates_posterior(fake_data, model)

        for method, df in zip(
            ["exposed_only", "difference", "posterior"],
            [exposed_only, difference, posterior],
        ):
            summary = get_summary(df)
            for key in summary:
                summaries[method][key].append(summary[key])

        if show_progress and i % 10 == 0:
            print(f"Completed repetition {i} of {num_repetitions - 1}")

    return summaries


def posterior_summary(model, data):
    fit = model.sample(data=data, adapt_delta=0.9, show_progress=False)
    return {
        "mu_theta": np.mean(fit.theta),
        "mu_b": np.mean(fit.b),
        "sigma_theta": np.std(np.mean(fit.theta, axis=0)),
        "sigma_b": np.std(np.mean(fit.b, axis=0)),
    }


def get_chick_data():
    chicks = pd.read_table("../nonadaptive/chickens.dat", sep="\\s+")
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
