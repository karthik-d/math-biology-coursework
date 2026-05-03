import numpy as np
import matplotlib.pyplot as plt

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

def simulate_gamblers_ruin(initial_wealth, N, p=0.5, max_steps=10000):
    """Simulates a single random walk for the gambler's ruin problem."""
    wealth = [initial_wealth]
    # Continue betting until ruin (0) or win (N)
    while wealth[-1] > 0 and wealth[-1] < N and len(wealth) < max_steps:
        step = 1 if np.random.rand() < p else -1
        wealth.append(wealth[-1] + step)
    return wealth

np.random.seed(42)

# ─── Part a: Sample Runs for Different N and Initial Conditions ────────────

N_values = [5, 10, 20]

# Dictionary defining [closer_to_0, closer_to_N] for each N
initial_conditions = {
    5: [2, 4],
    10: [4, 8],
    20: [6, 12]
}

# Create a 2x3 grid of subplots: Columns = N, Rows = Initial Condition
fig_a, axes_a = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
colors = plt.cm.tab10.colors

for col, N in enumerate(N_values):
    for row in range(2):
        ax = axes_a[row, col]
        initial_wealth = initial_conditions[N][row]
        
        # Simulate and plot 5 sample trajectories per panel
        for i in range(5):
            path = simulate_gamblers_ruin(initial_wealth, N, p=0.5)
            ax.plot(path, color=colors[i % len(colors)], alpha=0.7, linewidth=1.2)
            
        ax.axhline(N, color="green", linestyle="--", linewidth=1.5, label="Win ($N$)")
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Ruin ($0$)")
        
        # Titles and labels
        position_label = "Closer to 0" if row == 0 else "Closer to $N$"
        ax.set_title(f"$N = {N}$, Initial = ${initial_wealth}$ ({position_label})", 
                     fontweight="bold", fontsize=12)
        
        if row == 1:
            ax.set_xlabel("Bets (Time)")
        if col == 0:
            ax.set_ylabel("Wealth ($)")
            
        # Add legend only to the first subplot to prevent clutter
        if row == 0 and col == 0:
            ax.legend(loc="upper left")

fig_a.suptitle("Part a) Gambler's Ruin Sample Paths ($p = 1/2$)", 
               fontweight="bold", fontsize=15)
fig_a.tight_layout()


# ─── Part b: Likelihood of Winning (N=10) ──────────────────────────────────

N_b = 10
p_b = 0.5
num_trials = 5000
win_probs = []
initial_amounts = list(range(N_b + 1))

# Numerically estimate the win probability for each starting amount
for initial_wealth in initial_amounts:
    wins = 0
    for _ in range(num_trials):
        path = simulate_gamblers_ruin(initial_wealth, N_b, p=p_b)
        if path[-1] == N_b:
            wins += 1
    win_probs.append(wins / num_trials)

fig_b, ax_b = plt.subplots(figsize=(8, 5.5))

# Plot theoretical probability P(win) = i/N
theory_probs = [i / N_b for i in initial_amounts]
ax_b.plot(initial_amounts, theory_probs, color="#E84545", linewidth=2.5, 
          linestyle="--", label="Theoretical $P(\mathrm{Win}) = i/N$")

# Plot simulated probabilities as a scatter plot
ax_b.scatter(initial_amounts, win_probs, color="#20B2AA", s=80, 
             zorder=5, label=f"Simulated ($n={num_trials}$)", edgecolor="white")

# Calculate empirical error
error = np.mean(np.abs(np.array(win_probs) - np.array(theory_probs)))
stats_subtitle = f"Linear relationship confirmed | Mean Abs. Error approx {error:.4f}"

ax_b.set_title(f"Part b) Likelihood of Winning vs. Initial Amount ($N=10, p=1/2$)\n"
               rf"$\mathit{{{stats_subtitle}}}$", fontweight="bold", pad=10)
ax_b.set_xlabel("Initial Amount ($i$)")
ax_b.set_ylabel("Probability of Winning")
ax_b.set_xticks(initial_amounts)
ax_b.set_ylim(-0.05, 1.05)
ax_b.legend(loc="upper left", fontsize=11)

fig_b.tight_layout()

# Show both figures at once
plt.show()