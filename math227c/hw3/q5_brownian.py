import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F7F7F7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
})

sigma, T = 2.0, 10.0
CMAP = plt.cm.tab10

def sim_rw(dx, dt, T, n_paths=12, seed=42):
    """Random walk: X(t + Δt) = X(t) ± ΔX with prob 1/2."""
    np.random.seed(seed)
    steps = int(round(T / dt))
    t = np.linspace(0, T, steps + 1)
    paths = [
        np.r_[0, np.cumsum(np.random.choice([-dx, dx], size=steps))]
        for _ in range(n_paths)
    ]
    return t, paths


## Figure (a).

t_a, paths_a = sim_rw(dx=0.5, dt=1/3, T=T)

fig, ax = plt.subplots(figsize=(9, 5))
for i, x in enumerate(paths_a):
    ax.step(t_a, x, where="post",
            color=CMAP(i / 12), alpha=0.7, linewidth=1.3)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
ax.set_title("Part a)  —  Fixed ΔX = 0.5, Δt = 1/3\n"
             "12 sample paths of X(t) for t ∈ [0, 10]", fontweight="bold")
ax.set_xlabel("t")
ax.set_ylabel("X(t)")
fig.tight_layout()
plt.savefig("bm_part_a.png", dpi=150)
plt.show()


## Figure (b).

t_b1, paths_b1 = sim_rw(sigma * np.sqrt(0.10),  0.10,  T, seed=7)
t_b2, paths_b2 = sim_rw(sigma * np.sqrt(0.01),  0.01,  T, seed=7)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

for i, x in enumerate(paths_b1):
    ax1.plot(t_b1, x, color=CMAP(i / 12), alpha=0.65, linewidth=1.0)
ax1.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
ax1.set_title("Δt = 0.1  (ΔX = σ√Δt ≈ 0.632)", fontweight="bold")
ax1.set_xlabel("t");  ax1.set_ylabel("X(t)")

for i, x in enumerate(paths_b2):
    ax2.plot(t_b2, x, color=CMAP(i / 12), alpha=0.65, linewidth=1.0)
ax2.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
ax2.set_title("Δt = 0.01  (ΔX = σ√Δt = 0.2)", fontweight="bold")
ax2.set_xlabel("t");  ax2.set_ylabel("X(t)")

fig.suptitle("Part b)  —  Scaling Rule  ΔX = σ√Δt, σ = 2  |  "
             "Paths smoother as Δt → 0", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.savefig("bm_part_b.png", dpi=150)
plt.show()


## Figure (a).

dt_c   = 0.01
dx_c   = sigma * np.sqrt(dt_c)
steps_c = int(round(T / dt_c))   # 1000 steps per path
n_samp  = 8000

# Batch to avoid memory issues with large arrays
np.random.seed(99)
X_T = np.zeros(n_samp)
rng  = np.random.default_rng(99)
batch_size = 200
for start in range(0, n_samp, batch_size):
    end = min(start + batch_size, n_samp)
    inc = rng.choice([-dx_c, dx_c], size=(end - start, steps_c))
    X_T[start:end] = inc.sum(axis=1)

th_std  = sigma * np.sqrt(T)          # = 2√10 ≈ 6.32
x_range = np.linspace(-4*th_std, 4*th_std, 500)

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(X_T, bins=70, density=True, color="#20B2AA", alpha=0.70,
        edgecolor="white", linewidth=0.4, label="Simulated  X(10)")
ax.plot(x_range, norm.pdf(x_range, 0, th_std),
        color="#E84545", linewidth=2.5,
        label=f"$\\mathcal{{N}}(0,\\,\\sigma^2 T)$  — std = {th_std:.3f}")
ax.set_title(f"Part c)  —  Distribution of X(10)  vs  Theoretical Gaussian\n"
             f"Δt = {dt_c},  σ = {sigma},  T = {int(T)},  n = {n_samp} samples",
             fontweight="bold")
ax.set_xlabel("X(10)")
ax.set_ylabel("Probability Density")
ax.legend(fontsize=11)

# Annotate with numerical stats
stats_txt = (f"Simulated:  mean = {X_T.mean():.3f},  std = {X_T.std():.3f}\n"
             f"Theory:        mean = 0.000,  std = {th_std:.3f}")
ax.text(0.97, 0.95, stats_txt, transform=ax.transAxes, fontsize=10,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.9))
fig.tight_layout()
plt.savefig("bm_part_c.png", dpi=150)
plt.show()