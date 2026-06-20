import matplotlib.pyplot as plt
import numpy as np

from mcmc_calibrator.data_pipeline import calculate_hedge_ratio
from mcmc_calibrator.sampler import run_mcmc_calibrator


if __name__ == "__main__":
    ASSET_A = "GOOG"
    ASSET_B = "GOOGL"
    START = "2023-01-01"
    END = "2026-01-01"

    processed_df, hedge_ratio, intercept_val = calculate_hedge_ratio(
        ASSET_A,
        ASSET_B,
        START,
        END,
    )

    plt.figure(figsize=(12, 5))
    plt.plot(
        processed_df.index,
        processed_df["Spread"],
        label="Spread Series",
        color="#0369a1",
    )
    plt.axhline(
        processed_df["Spread"].mean(),
        color="red",
        linestyle="--",
        label="Spread Mean",
    )
    plt.title(f"Log Spread Series: {ASSET_A} - ({hedge_ratio:.3f} * {ASSET_B})")
    plt.xlabel("Date")
    plt.ylabel("Spread Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    spread_data = processed_df["Spread"].to_numpy()
    traces = run_mcmc_calibrator(spread_data, iterations=40000, dt=1.0)
    trace_theta, trace_mu, trace_sigma = traces

    burn_in = int(0.25 * len(trace_theta))
    clean_theta = trace_theta[burn_in:]
    clean_mu = trace_mu[burn_in:]
    clean_sigma = trace_sigma[burn_in:]

    calibrated_theta = np.median(clean_theta)
    calibrated_mu = np.median(clean_mu)
    calibrated_sigma = np.median(clean_sigma)

    print("\n================ CALIBRATION RESULTS ================")
    print(f"Speed of Reversion (Theta): {calibrated_theta:.4f}")
    print(f"Long-Term Mean     (Mu)   : {calibrated_mu:.4f}")
    print(f"Process Volatility (Sigma): {calibrated_sigma:.4f}")
    print(f"Expected Trading Half-Life: {np.log(2) / calibrated_theta:.2f} Days")
    print("=====================================================")
