import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Publication-style plot settings
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F7F7F7",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

# ─── Simulation Parameters ──────────────────────────────────────────────────
n_samples = 100000
alpha, beta, sigma = 0.5, 4.0, 2.0
t_eval = [1, 2, 3, 4, 5, 20]

np.random.seed(42)

# Initial distribution: Uniform(0, 1)
Z = np.random.uniform(0, 1, n_samples)
Z_history = {0: Z.copy()}

# Simulate forward in time vectorially
for t in range(1, 21):
    Z = alpha * Z + beta + np.random.normal(0, sigma, n_samples)
    if t in t_eval:
        Z_history[t] = Z.copy()

# ─── Part c: Evolution over time (t=1 and t=5) ──────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.hist(Z_history[1], bins=80, density=True, alpha=0.7, 
         color="#20B2AA", edgecolor="white", label="$t = 1$")
ax1.hist(Z_history[2], bins=80, density=True, alpha=0.7, 
         color="#FF7F50", edgecolor="white", label="$t = 2$")
ax1.hist(Z_history[3], bins=80, density=True, alpha=0.7, 
         color="green", edgecolor="white", label="$t = 3$")
ax1.hist(Z_history[4], bins=80, density=True, alpha=0.7, 
         color="purple", edgecolor="white", label="$t = 4$")

ax1.set_xlabel("$Z_t$")
ax1.set_ylabel("Probability Density")
ax1.set_title("Part c) Numerical Distribution of $Z_t$ over Time", fontweight="bold", pad=15)
ax1.legend(loc="upper right", fontsize=11, framealpha=0.9)
fig1.tight_layout()

# ─── Part d: Convergence to Stationary Distribution (t=20 vs Theory) ───────
fig2, ax2 = plt.subplots(figsize=(8, 5))

# Simulated histogram at t=20
ax2.hist(Z_history[20], bins=80, density=True, alpha=0.7, 
         color="#6495ED", edgecolor="white", label="Simulated $Z_{20}$")

# Theoretical stationary distribution N(8, 16/3)
th_mean = 8.0
th_var = 16 / 3
th_std = np.sqrt(th_var)
x_range = np.linspace(-1, 17, 400)

ax2.plot(x_range, norm.pdf(x_range, th_mean, th_std), 
         color="#E84545", linewidth=2.5, label=rf"Theory $\mathcal{{N}}(8, 16/3)$")

# Stats for subtitle
stats_subtitle = (f"Simulated mean: {Z_history[20].mean():.3f}, var: {Z_history[20].var():.3f}   |   "
                  f"Theory mean: {th_mean:.3f}, var: {th_var:.3f}")

ax2.set_xlabel("$Z_t$")
ax2.set_ylabel("Probability Density")
ax2.set_title(f"Part d) Convergence to Stationary Distribution $\pi$\n"
              rf"$\mathit{{{stats_subtitle}}}$", fontweight="bold", pad=10, fontsize=12)
ax2.legend(loc="upper right", fontsize=11, framealpha=0.9)
fig2.tight_layout()

# Show both plots
plt.show()