data {
    int<lower=1> num_expts;
    vector[num_expts] avg_treated_response;
    vector[num_expts] avg_control_response;
    vector[num_expts] treated_se;
    vector[num_expts] control_se;
    array[num_expts] int<lower=1, upper=num_expts> expt_id;
}
parameters {
    real mu_theta;
    real mu_b;
    real<lower=0> sigma_theta;
    real<lower=0> sigma_b;
    vector[num_expts] eta_theta;
    vector[num_expts] eta_b;
}
transformed parameters {
    vector[num_expts] theta;
    vector[num_expts] b;

    theta = mu_theta + sigma_theta * eta_theta;
    b = mu_b + sigma_b * eta_b;
}
model {
    eta_theta ~ normal(0, 1);
    eta_b ~ normal(0, 1);

    avg_treated_response ~ normal(b[expt_id] + theta[expt_id], treated_se);
    avg_control_response ~ normal(b[expt_id], control_se);
}