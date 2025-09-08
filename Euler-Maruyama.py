#GBM using Euler-Maruyama
import numpy as np
import matplotlib.pyplot as plt

def sidriftlate_gbm_euler(initial_price, drift, Vol, T, N, M=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    t = np.linspace(0, T, N+1)
    S = np.zeros((M, N+1))
    S[:, 0] = initial_price
    
    for n in range(N):
        dW = np.sqrt(dt) * np.random.randn(M)
        S[:, n+1] = S[:, n] + drift * S[:, n] * dt + vol * S[:, n] * dW
    
    return t, S

initial_price = 100
vol = 0.2
drift = 0.1
T = 1.0
r = 0.05
K = 100
N = 30600      # daily steps
M = 5          # number of paths

t, S = sidriftlate_gbm_euler(initial_price, drift, vol, T, N, M, seed=42)

# Plot
plt.figure(figsize=(10,6))
for m in range(M):
    plt.plot(t, S[m], lw=1)
plt.title("Geometric Brownian Motion (Euler–Maruyama)")
plt.xlabel("Time (years)")
plt.ylabel("Stock Price")
plt.grid(True)
plt.show()