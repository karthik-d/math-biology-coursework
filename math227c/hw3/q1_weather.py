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

# Define transition probabilities (adjust alpha/beta to match your specific notes)
alpha, beta = 0.4, 0.2

# Transition matrix P: 
# Row 0: Rainy -> [Rainy, Sunny]
# Row 1: Sunny -> [Rainy, Sunny]
P = np.array([
    [1 - alpha, alpha],
    [beta,      1 - beta]
])

print("Transition Matrix P:")
print(P)
print("-" * 40)

# ─── Part c: Calculate numerically the probability at t = 3 ───────────────

# Initial state: Sunny at t=0
# pi_0 = [P(Rainy), P(Sunny)]
pi_0 = np.array([0.0, 1.0])

# Compute P^3 directly
P_3 = np.linalg.matrix_power(P, 3)

# State at t=3 is given by pi_3 = pi_0 @ P^3
pi_3 = pi_0 @ P_3

# The probability that it is rainy at t=3 is the first component
prob_rainy_t3 = pi_3[0]

print("Part c)")
print(f"Matrix P^3:\n{P_3}")
print(f"Distribution at t=3: {pi_3}")
print(f"P(Rainy at t=3 | Sunny at t=0) = {prob_rainy_t3:.4f}")
print("-" * 40)

# ─── Part d: Illustrate convergence towards the stationary distribution ────

# Arbitrary initial distribution (e.g., 80% Rainy, 20% Sunny)
pi_current = np.array([0.8, 0.2])

# Theoretical stationary distribution derived in Part b
pi_stat = np.array([beta / (alpha + beta), alpha / (alpha + beta)])

n_steps = 15
history = [pi_current]

# Iterate the system
for t in range(n_steps):
    pi_current = pi_current @ P
    history.append(pi_current)

history = np.array(history)

# Plot the evolution of the probabilities
fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(range(n_steps + 1), history[:, 0], 'o-', color="#20B2AA", 
        linewidth=2, label=r'$\pi_t(\mathrm{Rainy})$')
ax.plot(range(n_steps + 1), history[:, 1], 's-', color="#FF7F50", 
        linewidth=2, label=r'$\pi_t(\mathrm{Sunny})$')

# Add theoretical stationary limits as dashed reference lines
ax.axhline(pi_stat[0], color="#20B2AA", linestyle='--', alpha=0.6, 
           label=rf'Stationary $\pi_1 = {pi_stat[0]:.3f}$')
ax.axhline(pi_stat[1], color="#FF7F50", linestyle='--', alpha=0.6, 
           label=rf'Stationary $\pi_2 = {pi_stat[1]:.3f}$')

ax.set_xlabel("Time step ($t$)")
ax.set_ylabel("Probability")
ax.set_title("Part d) Convergence to Stationary Distribution", fontweight="bold", pad=15)
ax.set_ylim(0, 1.05)

# Place legend cleanly
ax.legend(loc="center right", fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.show()