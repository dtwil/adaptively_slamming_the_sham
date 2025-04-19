data {
    int<lower=1> J;
    vector[J] y1;
    vector[J] y0;
    vector<lower=0>[J] sigma1;
    vector<lower=0>[J] sigma0;
    array[J] int<lower=1, upper=J> j;

    // The following are the known hyperparameters
    real mu_theta;
    real<lower=0> sigma_theta;
    real mu_b;
    real<lower=0> sigma_b;
}
parameters {
    vector[J] eta_theta;
    vector[J] eta_b;
}
transformed parameters {
    vector[J] theta;
    vector[J] b;

    theta = mu_theta + sigma_theta * eta_theta;
    b = mu_b + sigma_b * eta_b;
}
model {
    eta_theta ~ normal(0, 1);
    eta_b ~ normal(0, 1);

    y1 ~ normal(b + theta, sigma1);
    y0 ~ normal(b, sigma0);
}