data {
    int<lower=1> J;
    vector[J] y1_bar;
    vector[J] y0_bar;
    vector<lower=0>[J] sigma_y1_bar;
    vector<lower=0>[J] sigma_y0_bar;
    array[J] int<lower=1, upper=J> j;
}
parameters {
    real mu_theta;
    real mu_b;
    real<lower=0> sigma_theta;
    real<lower=0> sigma_b;
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

    y1_bar ~ normal(b + theta, sigma_y1_bar);
    y0_bar ~ normal(b, sigma_y0_bar);
}