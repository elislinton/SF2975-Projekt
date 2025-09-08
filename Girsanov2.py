# Simulate under P, reweight to Q using Radon-Nikodym derivative, and compare to direct Q-simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, exp

# Parameters
S0 = 100.0
mu = 0.08       # real-world drift
r = 0.03        # risk-free rate
sigma = 0.2
T = 1.0         # 1 year
K = 100.0       # strike
N = 200_000     # number of Monte Carlo samples

# Seed for reproducibility
np.random.seed(42)

# Market price of risk
theta = (mu - r) / sigma

# 1) Simulate terminal Brownian under P (W_T ~ N(0, T))
W_T_P = np.sqrt(T) * np.random.randn(N)

# Terminal asset price under P (closed-form GBM)
log_S_T_P = np.log(S0) + (mu - 0.5 * sigma**2) * T + sigma * W_T_P
S_T_P = np.exp(log_S_T_P)

# Radon-Nikodym derivative Z_T = exp(-theta W_T - 0.5 theta^2 T)
Z_T = np.exp(-theta * W_T_P - 0.5 * theta**2 * T)

# Payoff under P sample
payoff_P = np.maximum(S_T_P - K, 0.0)

# Risk-neutral expectation via reweighting:
# E_Q[payoff] = E_P[payoff * Z_T]
price_reweighted = np.exp(-r * T) * np.mean(payoff_P * Z_T)
se_reweighted = np.exp(-r * T) * np.std(payoff_P * Z_T, ddof=1) / np.sqrt(N)

# Diagnostics
mean_Z = np.mean(Z_T)
var_Z = np.var(Z_T, ddof=1)
ess = (np.sum(Z_T)**2) / np.sum(Z_T**2)  # effective sample size (unnormalized by N)
ess_rel = ess / N  # relative ESS

# 2) Direct simulation under Q (drift r)
W_T_Q = np.sqrt(T) * np.random.randn(N)
log_S_T_Q = np.log(S0) + (r - 0.5 * sigma**2) * T + sigma * W_T_Q
S_T_Q = np.exp(log_S_T_Q)
payoff_Q = np.maximum(S_T_Q - K, 0.0)

price_directQ = np.exp(-r * T) * np.mean(payoff_Q)
se_directQ = np.exp(-r * T) * np.std(payoff_Q, ddof=1) / np.sqrt(N)

#P_payoff = np.mean(payoff_P * Z_T)
#print("P-payoff:" P_payoff)

# Results summary table
results = pd.DataFrame({
    "method": ["Reweighted (simulate P & weight)", "Direct Q-simulation"],
    "discounted_price": [price_reweighted, price_directQ],
    "std_error": [se_reweighted, se_directQ],
    "N": [N, N]
})

# Print diagnostic info and show small table
print(f"Parameters: S0={S0}, mu={mu}, r={r}, sigma={sigma}, T={T}, K={K}, N={N}")
print(f"theta (market price of risk) = {theta:.6f}")
print(f"Mean Z_T (should be ~1) = {mean_Z:.6f}, Var(Z_T) = {var_Z:.6e}")
print(f"ESS = {ess:.1f} (relative ESS = {ess_rel:.4f})\n")
print(results.to_string(index=False))

# Plot distributions of terminal prices and weights (random sample for plotting)
sample_idx = np.random.choice(N, size=2000, replace=False)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(S_T_P[sample_idx], bins=40)
plt.title("Subset: S_T under P (sample)")

plt.subplot(1,2,2)
plt.hist(Z_T[sample_idx], bins=40)
plt.title("Subset: RN weights Z_T (sample)")
plt.tight_layout()
plt.show()

# Show numeric values as DataFrame for the user
#import caas_jupyter_tools as cjt
#cjt.display_dataframe_to_user("Monte Carlo Estimates", results)

# Also display a few sample rows (S_T, payoff, Z) to illustrate
df_sample = pd.DataFrame({
    "S_T_P (first 10)": S_T_P[:10],
    "payoff_P (first 10)": payoff_P[:10],
    "Z_T (first 10)": Z_T[:10]
})
#cjt.display_dataframe_to_user("Sample of simulated paths under P (first 10 rows)", df_sample)
