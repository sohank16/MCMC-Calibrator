import numpy as np

from mcmc_calibrator.ou_model import compute_log_likelihood


def run_mcmc_calibrator(data, iterations=30000, dt=1.0):
    print(f"[*] Initializing MCMC Calibrator for {iterations} iterations...")

    current_theta = 1.0
    current_mu = data.mean()
    current_sigma = data.std()

    current_log_lik = compute_log_likelihood(
        data,
        current_theta,
        current_mu,
        current_sigma,
        dt,
    )

    trace_theta = np.zeros(iterations)
    trace_mu = np.zeros(iterations)
    trace_sigma = np.zeros(iterations)

    proposal_widths = {"theta": 0.005, "mu": 0.00025, "sigma": 0.00025}

    accepted_count = 0

    for i in range(iterations):
        proposed_theta = current_theta + np.random.normal(
            0,
            proposal_widths["theta"],
        )
        proposed_mu = current_mu + np.random.normal(0, proposal_widths["mu"])
        proposed_sigma = current_sigma + np.random.normal(
            0,
            proposal_widths["sigma"],
        )

        proposed_log_lik = compute_log_likelihood(
            data,
            proposed_theta,
            proposed_mu,
            proposed_sigma,
            dt,
        )

        log_acceptance_ratio = proposed_log_lik - current_log_lik

        if np.log(np.random.uniform(0, 1)) < log_acceptance_ratio:
            current_theta = proposed_theta
            current_mu = proposed_mu
            current_sigma = proposed_sigma
            current_log_lik = proposed_log_lik
            accepted_count += 1

        trace_theta[i] = current_theta
        trace_mu[i] = current_mu
        trace_sigma[i] = current_sigma

    acceptance_rate = (accepted_count / iterations) * 100
    print(f"[+] Simulation Complete. Global Acceptance Rate: {acceptance_rate:.2f}%")

    return trace_theta, trace_mu, trace_sigma
