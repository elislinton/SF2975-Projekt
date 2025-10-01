import numpy as np

def monte_carlo_option_pricing(S0, K, T, r, sigma, n_sim=100000):
    # Simulate end-of-period stock prices
    np.random.seed(10)
    Z = np.random.randn(n_sim)  # standard normal random draws
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Calculate payoffs
    call_payoffs = np.maximum(ST - K, 0)
    put_payoffs  = np.maximum(K - ST, 0)

    # Discount back
    disc_call = np.exp(-r * T) * call_payoffs
    disc_put  = np.exp(-r * T) * put_payoffs

    # Means
    call_price = np.mean(disc_call)
    put_price = np.mean(disc_put)

    return call_price, put_price

S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
call, put = monte_carlo_option_pricing(S0, K, T, r, sigma)

put_parity = call + K * np.exp(-r * T) - S0

print(f"European Put Price: {put:.4f}")
print(f"Put Price from put call parity: {put_parity:.4f}")