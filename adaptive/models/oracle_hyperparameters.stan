data {
    int<lower=1> J;
    vector[J] y1_bar;
    vector[J] y0_bar;
    vector<lower=0>[J] sigma_y1_bar;
    vector<lower=0>[J] sigma_y0_bar;
    array[J] int<lower=1, upper=J> j;

    // The following are the known hyperparameters
    real mu_theta;
    real<lower=0> sigma_theta;
    real mu_b;
    real<lower=0> sigma_b;
}
parameters {
    vector[J] theta;
    vector[J] b;
}
model {
    theta ~ normal(mu_theta, sigma_theta);
    b ~ normal(mu_b, sigma_b);

    y1_bar ~ normal(b + theta, sigma_y1_bar);
    y0_bar ~ normal(b, sigma_y0_bar);
}