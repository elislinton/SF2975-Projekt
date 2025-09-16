import numpy as np
import matplotlib.pyplot as plt 
from Girsanov import simulate_gbm_paths

# Calculate Radon-Nikodym derivative Z_T
def reweight_paths(paths, mu, r, sigma, dt):
    n_steps, _ = paths.shape[0]-1, paths.shape[1]
    T = n_steps * dt
    
    # Browian motion
    ST = paths[-1]
    S0 = paths[0,0]
    WT = (np.log(ST/S0) - (mu - 0.5*sigma**2)*T) / sigma
    theta = (mu - r)/sigma
  
    #Derivative
    Z = np.exp(-theta*WT - 0.5*theta**2*T)
    return Z

# Example usage with given parameters
def example():
    # Parameters
    mu, sigma, r = 0.1, 0.2, 0.05
    S0, K = 100, 100
    T, dt = 1, 1/252 # 1 year, 252 trading days
    n_paths =  100

    # Seed for reproducability
    np.random.seed(10)

    # Simulate under P
    paths_P = simulate_gbm_paths(S0, mu, r, sigma, T, dt, n_paths, measure="P")
    ST = paths_P[-1]
    # Calculate RN-derivative
    Z = reweight_paths(paths_P, mu, r, sigma, dt)

    # Risk neural price of call given by reweight
    payoff = np.maximum(ST - K, 0)
    price_via_reweight = np.exp(-r*T) * np.mean(payoff * Z)

    # Calculate price under Q for comparison
    paths_Q = simulate_gbm_paths(S0, mu, r, sigma, T, dt, n_paths, measure="Q")
    ST_Q = paths_Q[-1]
    price_via_Q = np.exp(-r*T) * np.mean(np.maximum(ST_Q - K, 0))

    print("Price via reweighting under P:", price_via_reweight)
    print("Price via direct Q-simulation:", price_via_Q)


# Sweep number of paths to show convergence to Q-price
def n_path_sweep():
    path_list = np.linspace(100, 100000, 100)
    # Parameters
    mu, sigma, r = 0.1, 0.2, 0.05
    S0, K = 100, 100
    T, dt = 1, 1/252
    Prices_P = []
    Prices_Q = []

    for n_paths in path_list:
        # Seed for reproducability
        np.random.seed(10)

        # Simulate under P
        paths_P = simulate_gbm_paths(S0, mu, r, sigma, T, dt, int(n_paths), measure="P")
        ST = paths_P[-1]
        Z = reweight_paths(paths_P, mu, r, sigma, dt)

        # Risk neural price of call given by reweight
        payoff = np.maximum(ST - K, 0)
        price_via_reweight = np.exp(-r*T) * np.mean(payoff * Z)
        Prices_P.append(price_via_reweight)

        # Calculate price under Q for comparison
        paths_Q = simulate_gbm_paths(S0, mu, r, sigma, T, dt, int(n_paths), measure="Q")
        ST_Q = paths_Q[-1]
        price_via_Q = np.exp(-r*T) * np.mean(np.maximum(ST_Q - K, 0))
        Prices_Q.append(price_via_Q)

    plt.plot(path_list, Prices_P, label='P-price');plt.plot(path_list, Prices_Q, label='Q-price')
    plt.xlabel("Number of paths");plt.ylabel("Price");plt.legend();plt.show()


if __name__== "__main__":
    example()
