# MCMC Calibrator

This is a simple Python version of the Stamatics MCMC calibrator project.

The project flow is:

```text
download prices
-> calculate hedge ratio
-> build spread
-> check stationarity with ADF
-> run OU log-likelihood
-> run Metropolis-Hastings MCMC
-> estimate theta, mu, sigma, and half-life
```

## Folder Structure

```text
.
├── main.py
├── requirements.txt
└── mcmc_calibrator/
    ├── __init__.py
    ├── data_pipeline.py
    ├── ou_model.py
    └── sampler.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

By default, the script uses `GOOG` and `GOOGL` from `2023-01-01` to `2026-01-01`, matching the current notebook direction.

To try another pair, edit these values inside `main.py`:

```python
ASSET_A = "GOOG"
ASSET_B = "GOOGL"
START = "2023-01-01"
END = "2026-01-01"
```

## Main Idea

We first create a spread:

```text
Spread = log(Asset A) - beta * log(Asset B)
```

Then we model that spread as an Ornstein-Uhlenbeck process and use MCMC to estimate:

- `theta`: speed of mean reversion
- `mu`: long-term mean
- `sigma`: process volatility
- `half-life`: expected time for the spread deviation to reduce by half

## Contributors

Neelatmajam Dwivedi - 240687  
Aditya Kukreti - 240058  
Palash Lahoti - 240721  
Prithvijeet Bhattacharya - 240799  
Kadam Sohan Santosh - 240506  
Ved Kartikey - 241150  
Kiran Khandu - 240547  


