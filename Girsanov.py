import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm_paths(S0, mu, r, sigma, T, dt, n_paths, measure="P"):
    """
    Simulate GBM paths under P (real-world) or Q (risk-neutral).
    
    S0      : initial asset price
    mu      : drift under P
    r       : risk-free rate
    sigma   : volatility
    T       : time horizon
    dt      : time step
    n_paths : number of simulated paths
    measure : "P" (real-world) or "Q" (risk-neutral)
    """
    n_steps = int(T / dt)
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = S0

    drift = mu if measure == "P" else r

    for t in range(1, n_steps + 1):
        Z = np.random.normal(0, 1, n_paths)
        paths[t] = paths[t-1] * np.exp(
            (drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    return paths

# Example usage
S0, mu, r, sigma, T, dt, n_paths = 100, 0.08, 0.03, 0.2, 1, 1/252, 5

paths_P = simulate_gbm_paths(S0, mu, r, sigma, T, dt, n_paths, measure="P")
paths_Q = simulate_gbm_paths(S0, mu, r, sigma, T, dt, n_paths, measure="Q")

# Plot results
plt.figure(figsize=(10,5))
plt.plot(paths_P, alpha=0.7)
plt.title("Simulated Asset Price Paths under Real-World Measure (P)")
plt.xlabel("Time step")
plt.ylabel("Price")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(paths_Q, alpha=0.7)
plt.title("Simulated Asset Price Paths under Risk-Neutral Measure (Q)")
plt.xlabel("Time step")
plt.ylabel("Price")
plt.show()
