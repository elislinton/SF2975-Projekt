import matplotlib.pyplot as plt
import numpy as np
from Girsanov import simulate_gbm_paths
from scipy import stats
from scipy.stats import norm


n_paths, n_points   = 5, [1000]     # antal paths per körning samt antal punkter, kan ta flera samtidigt för jämförelse
mu, sigma, r, q     = 0.1, 0.2, 0.05, 0 # fördelningsparametrar, utdelning q antas 0
S_0, K,  W          = 100, 100, 0.0
T                   = 1
interval            = [0, T]
frequencies         = list(range(1,51))              # frekvenser Eför ombalanser, e.g. 1 ombalanserar varje tidssteg, 5 ombalanserar var femte tidssteg

# Funktion för att beräkna call pris med BS
def bs_call_price(t, S_t):
    tau = T-t
    d_1 = (np.log(S_t/K) + (r-q+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    d_2 = d_1 - sigma*np.sqrt(tau)
    return S_t*np.exp(-q*tau)*norm.cdf(d_1) - K*np.exp(-r*tau)*norm.cdf(d_2)

# Funtkion för delta av en call vid viss tidpunkt
def delta_call(t, S_t):
    tau = T-t
    d_1 = (np.log(S_t/K) + (r-q+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    return np.exp(-q*tau)*norm.cdf(d_1)

# Main loop, backtracking över tidsintervallet för att beräkna hedging i varje steg
std_list = []
for n in n_points:
        dt = (interval[1] - interval[0]) / (n-1)
        t_axis = np.linspace(interval[0], interval[1], n)

        # simulera under Q
        path = simulate_gbm_paths(S_0, mu, r, sigma, T, dt, n_paths, measure='Q')
        S_T = path[-1]
        payoff_C = np.maximum(S_T - K, 0.0)

        for freq in frequencies:
            # --- delta-hedge med UNDERLIGGANDE, självfinansierad ---
            # start: optionens pris och delta vid t0 på varje path
            Delta = delta_call(0.0, path[0, :])                 # (n_paths,)
            V0    = bs_call_price(0.0, path[0, :])              # (n_paths,)
            B     = V0 - Delta * path[0]                        # vi håller Delta aktier -> kassakonto som resten

            # rebalansera varje steg med vänsterpunkts-delta
            for k in range(1, n-1):
                B *= np.exp(r * dt)                           # väx kontant
                if k%freq == 0:
                    D_new = delta_call(t_axis[k], path[k])        # ny delta vid t_k
                    B -= (D_new - Delta) * path[k]                # köp/sälj aktier; självfinansierat
                    Delta = D_new

            B *= np.exp(r*dt)  # sista räntepåslag till T
            V_T = Delta * S_T + B
            hedgingError  = V_T - payoff_C
            print(f"freq = {freq}")
            #print(f"Portfolio end value = {(np.round(V_T, 3))}")
            print(f"n={n}  mean(hedgingError)={hedgingError.mean():.6f}  std(hedgingError)={hedgingError.std():.6f}")
            std_list.append(hedgingError.std())


# plot av std för hedging error över ombalansfrekvenser
slope, intercept, _, _, _ = stats.linregress(frequencies, std_list)
y_fit = slope*np.array(frequencies) + intercept
plt.figure(figsize=(10,5))
plt.plot(frequencies,std_list, label="std values")
plt.plot(frequencies,y_fit, label=f"regression (slope={slope:.4f})")
plt.title("Standard deviations of hedging error over rebalancing frequency")
plt.xlabel("Frequency")
plt.ylabel("Standard deviation")
plt.tight_layout()
plt.legend()
plt.show()

