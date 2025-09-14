# Task b
import numpy as np

def monte_carlo_option_pricing(S0, K, T, r, sigma, n_sim=100000):
    """
    Monte Carlo simulation for European call and put option pricing.
    
    Parameters:
        S0    : initial stock price
        K     : strike price
        T     : time to maturity (in years)
        r     : risk-free interest rate
        sigma : volatility
        n_sim : number of simulations
    
    Returns:
        call_price, put_price
    """
    # Simulate end-of-period stock prices
    Z = np.random.randn(n_sim)  # standard normal random draws
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Calculate payoffs
    call_payoffs = np.maximum(ST - K, 0)
    put_payoffs  = np.maximum(K - ST, 0)

    # Discount back
    call_price = np.exp(-r * T) * np.mean(call_payoffs)
    put_price  = np.exp(-r * T) * np.mean(put_payoffs)


    return call_price, put_price


# Example usage
S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
call, put = monte_carlo_option_pricing(S0, K, T, r, sigma)

print(f"European Call Price (MC): {call:.4f}")

print(f"European Put Price (MC):  {put:.4f}")
