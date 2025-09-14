# Task f)
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Normal CDF using error function ---
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# --- Black-Scholes call price ---
def bs_call_price(S0, K, T, r, sigma):
    d1 = (math.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S0*norm_cdf(d1) - K*math.exp(-r*T)*norm_cdf(d2)

# --- Implied volatility via bisection ---
def implied_volatility_call(price, S0, K, T, r, tol=1e-6, max_iter=100):
    low, high = 1e-6, 5.0  # search interval for vol
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        price_mid = bs_call_price(S0, K, T, r, mid)
        if abs(price_mid - price) < tol:
            return mid
        if price_mid > price:
            high = mid
        else:
            low = mid
    return mid  # return best guess if no exact root

# --- Parameters ---
S0, K, T, r = 100, 100, 1.0, 0.05

# --- Vary volatility ---
vols = np.linspace(0.01, 1.0, 100)
call_prices = [bs_call_price(S0, K, T, r, sigma) for sigma in vols]

# --- Plot Call price vs Volatility ---
plt.figure(figsize=(8,5))
plt.plot(vols, call_prices, label="Call Price")
plt.xlabel("Volatility (sigma)")
plt.ylabel("Call Price")
plt.title("European Call Price as a function of Volatility")
plt.legend()
plt.grid(True)
plt.show()

# --- Example: compute implied vols for given prices ---
given_prices = [10, 15, 20]
implied_vols = [implied_volatility_call(p, S0, K, T, r) for p in given_prices]

for price, iv in zip(given_prices, implied_vols):
    print(f"Option price = {price:.2f}  ->  Implied Vol = {iv:.4f}")
