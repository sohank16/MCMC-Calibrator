import numpy as np


def compute_log_likelihood(data, theta, mu, sigma, dt=1.0):
    """Evaluate the log-likelihood of OU parameters for observed spread data."""
    if theta <= 0 or sigma <= 0:
        return -np.inf

    x_old = data[:-1]
    x_new = data[1:]
    n = len(x_old)

    exp_term = np.exp(-theta * dt)
    conditional_mean = x_old * exp_term + mu * (1.0 - exp_term)
    conditional_var = (sigma**2 / (2.0 * theta)) * (
        1.0 - np.exp(-2.0 * theta * dt)
    )

    residuals = x_new - conditional_mean
    sum_squared_residuals = np.sum(residuals**2)

    log_likelihood = -0.5 * n * np.log(2.0 * np.pi * conditional_var) - (
        sum_squared_residuals / (2.0 * conditional_var)
    )
    return log_likelihood
