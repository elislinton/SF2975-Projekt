# Least-Squares Monte Carlo (Longstaff-Schwartz) for American put option pricing
# The code will simulate paths under risk-neutral GBM, then use least-squares regression
# on in-the-money paths to estimate continuation values and decide early exercise.
# It returns the estimated American put price and the European put price (Black-Scholes) for comparison.

import numpy as np
from math import log, sqrt, exp, erf
np.random.seed(42)

def normal_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def black_scholes_put(S, K, T, r, sigma):
    # Closed-form Black-Scholes put price (European)
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)

def lsm_american_put_price(S0, K, T, r, sigma, n_sim=100000, n_steps=50, basis_degree=2):
    """
    Longstaff-Schwartz LSM for American put.
    - basis_degree: use polynomial basis up to this power (1 => [1, S], 2 => [1, S, S^2], ...)
    Returns estimated option price.
    """
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # Simulate asset paths (vectorized). Use standard normal increments.
    Z = np.random.randn(n_sim, n_steps)
    # Optionally use antithetic sampling to reduce variance (here we don't to keep clarity)
    
    # Build log-returns and simulate S paths
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    logS = np.log(S0) + increments.cumsum(axis=1)
    S = np.exp(logS)
    # Prepend S0 as the t=0 column
    S = np.hstack((np.full((n_sim,1), S0), S))
    # Time indices 0...n_steps, total n_steps+1 times
    times = np.linspace(0, T, n_steps+1)
    
    # Payoffs at all times for put: max(K - S, 0)
    payoffs = np.maximum(K - S, 0.0)
    
    # Cashflows: initialize with payoff at maturity (time index n_steps)
    cashflow = payoffs[:, -1].copy()
    # We will work backwards in time to determine exercise
    for t in range(n_steps-1, 0, -1):  # from n_steps-1 down to 1 (we don't exercise at time 0)
        # find in-the-money paths at time t
        itm = payoffs[:, t] > 0
        if not np.any(itm):
            # no in-the-money paths this time step
            # discount cashflows and continue
            cashflow = cashflow * discount
            continue
        
        # Prepare regression: X are basis functions of S_t for itm paths
        St_itm = S[itm, t]
        # Build design matrix with columns [1, S, S^2, ... up to basis_degree]
        X = np.vstack([St_itm**d for d in range(basis_degree+1)]).T  # shape: (n_itm, basis_degree+1)
        
        # The dependent variable Y is the discounted cashflow from "continuation" (one step ahead)
        Y = cashflow[itm] * discount  # discount one step back
        
        # Solve least squares: coeffs = argmin ||X beta - Y||^2
        # Use np.linalg.lstsq for numerical stability
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        
        # Estimate continuation values for itm paths
        cont_est = X.dot(beta)
        
        # Immediate exercise payoff
        immediate_payoff = payoffs[itm, t]
        
        # For those paths where immediate_payoff > continuation estimate, exercise now
        exercise = immediate_payoff > cont_est
        # Update cashflows: for exercised paths, set cashflow to immediate payoff;
        # for non-exercised, cashflow remains (already representing later exercise) but we need to discount afterwards
        # Create a mask for global paths that are exercised now:
        exercised_idx = np.where(itm)[0][exercise]  # convert relative indices to global indices
        cashflow[exercised_idx] = immediate_payoff[exercise]  # set to payoff at time t
        
        # Discount cashflow for the next iteration (moving one step back in time)
        cashflow = cashflow * discount
    
    # After finishing, price is the expected discounted cashflow at t=0:
    price_estimate = np.mean(cashflow)  # cashflow already discounted all the way to time 0
    # Standard error estimate:
    stderr = np.std(cashflow, ddof=1) / np.sqrt(n_sim)
    
    return price_estimate, stderr

# Example parameters (classic test case often used in literature)
S0 = 100.0
K  = 100.0
T  = 1.0
r  = 0.05
sigma = 0.2

# Run LSM
n_sim = 120000   # number of simulated paths
n_steps = 50     # time steps
price, stderr = lsm_american_put_price(S0, K, T, r, sigma, n_sim=n_sim, n_steps=n_steps, basis_degree=2)

# European put (Black-Scholes) for comparison
european_put = black_scholes_put(S0, K, T, r, sigma)

print(price, stderr, european_put)

