import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import yfinance as yf


def calculate_hedge_ratio(ticker_a, ticker_b, start_date, end_date):
    """Download data, calculate hedge ratio, build spread, and run ADF test."""
    print(f"[*] Fetching historical data for {ticker_a} and {ticker_b}...")

    raw_data = yf.download(
        [ticker_a, ticker_b],
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    data = pd.DataFrame()
    data[ticker_a] = raw_data["Close"][ticker_a]
    data[ticker_b] = raw_data["Close"][ticker_b]
    data = data.dropna()

    log_a = np.log(data[ticker_a])
    log_b = np.log(data[ticker_b])

    X = sm.add_constant(log_b)
    model = sm.OLS(log_a, X).fit()

    alpha = model.params["const"]
    beta = model.params[ticker_b]

    print("\n================ OLS REGRESSION SUMMARY ================")
    print(f"Hedge Ratio (Beta): {beta:.6f}")
    print(f"Intercept  (Alpha): {alpha:.6f}")
    print(f"R-squared Value   : {model.rsquared:.4f}")
    print("========================================================\n")

    data["Spread"] = log_a - (beta * log_b)

    adf_result = adfuller(data["Spread"])

    print("================== STATIONARITY CHECK ==================")
    print(f"ADF Test Statistic: {adf_result[0]:.4f}")
    print(f"p-value            : {adf_result[1]:.4f}")
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value:.4f}")

    if adf_result[1] < 0.05:
        print("\n[SUCCESS] The spread is STATIONARY (p < 0.05).")
        print("This pair is structurally valid for MCMC Calibration.")
    else:
        print("\n[WARNING] The spread is NON-STATIONARY (p >= 0.05).")
        print("MCMC parameter derivation may capture unstable random walks.")
    print("========================================================\n")

    return data, beta, alpha
