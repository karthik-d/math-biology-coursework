import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Parameters
sigma = 0.2
dt = 0.001
t_max = 20.0
n_steps = int(t_max / dt)
t = np.linspace(0, t_max, n_steps)

# 2. Define simulation function
def f(x):
    """Deterministic drift function."""
    return -x * (x - 1) * (x - 2)

def simulate_sde(x0, n_steps, dt, sigma):
    """Simulate SDE using the Euler-Maruyama method."""
    x = np.zeros(n_steps)
    x[0] = x0
    
    # Generate Gaussian noise for the entire trajectory: N(0, dt)
    dW = np.sqrt(dt) * np.random.randn(n_steps - 1)
    
    for i in range(n_steps - 1):
        x[i+1] = x[i] + f(x[i]) * dt + sigma * dW[i]
    return x

# 3. Run simulations
np.random.seed(42) # For reproducible trajectories
x_09 = simulate_sde(0.9, n_steps, dt, sigma)
x_11 = simulate_sde(1.1, n_steps, dt, sigma)

# 4. Plotting
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Stochastic Time Evolution: $dX = -X(X-1)(X-2)dt + 0.2dB_t$", fontsize=16)

# Plot trajectories with distinct colors and slight transparency
ax.plot(t, x_09, color='#1f77b4', linewidth=1.2, alpha=0.85, label='Trajectory from $X_0 = 0.9$')
ax.plot(t, x_11, color='#ff7f0e', linewidth=1.2, alpha=0.85, label='Trajectory from $X_0 = 1.1$')

# Add steady state reference lines with clear distinctions
ax.axhline(0, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7, label='Stable Equilibrium ($x^*=0$)')
ax.axhline(1, color='darkred', linestyle='-.', linewidth=2, alpha=0.7, label='Unstable Equilibrium ($x^*=1$)')
ax.axhline(2, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7, label='Stable Equilibrium ($x^*=2$)')

ax.set_title('Euler-Maruyama Simulation ($\\Delta t = 0.001$)', fontsize=14)
ax.set_xlabel('Time ($t$)', fontsize=13)
ax.set_ylabel('State $X(t)$', fontsize=13)

# Improve legend placement outside the main trace area
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
ax.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()