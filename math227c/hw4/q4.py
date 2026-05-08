import numpy as np
import matplotlib.pyplot as plt

# 1. Set Parameters
alpha = 0.8
lam = 2.0
n_steps = 10000
burn_in = 1000

# Theoretical values from Part c calculations
theo_mean = 1.0 / (lam * (1 - alpha))
theo_var = 1.0 / ((lam**2) * (1 - alpha**2))

# 2. Run Simulation
np.random.seed(42)  # For reproducibility
Z = np.zeros(n_steps)
Z[0] = 0.0 # Initial state

# Generate exponential noise (Note: numpy exponential uses scale = 1/lambda)
eps = np.random.exponential(scale=1.0/lam, size=n_steps)

for t in range(1, n_steps):
    Z[t] = alpha * Z[t-1] + eps[t]

# 3. Analyze Stationary Distribution
# Discard the transitional burn-in period
Z_stat = Z[burn_in:]

emp_mean = np.mean(Z_stat)
emp_var = np.var(Z_stat)

# 4. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(f'AR(1) Process with Exponential Noise ($\\alpha={alpha}$, $\\lambda={lam}$)', fontsize=16, y=1.02)

# Plot 1: Trace plot (first 500 steps to show burn-in)
ax1.plot(range(500), Z[:500], color='#1f77b4', linewidth=1.2, alpha=0.85)
ax1.axvspan(0, 100, color='gray', alpha=0.2, label='Rapid convergence zone')
ax1.axhline(y=theo_mean, color='red', linestyle='--', linewidth=2, label=f'Theoretical Mean = {theo_mean:.2f}')

ax1.set_title('Trace Plot (First 500 steps)', fontsize=14)
ax1.set_xlabel('Time Step (t)', fontsize=12)
ax1.set_ylabel('State $Z_t$', fontsize=12)
ax1.legend(loc='lower right')
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Histogram of stationary distribution
ax2.hist(Z_stat, bins=50, density=True, color='#1f77b4', edgecolor='white', alpha=0.7, label='Empirical Density')
ax2.axvline(x=emp_mean, color='green', linestyle='-', linewidth=2.5, 
            label=f'Emp. Mean: {emp_mean:.2f}\nEmp. Var: {emp_var:.2f}')
ax2.axvline(x=theo_mean, color='red', linestyle='--', linewidth=2.5, 
            label=f'Theo. Mean: {theo_mean:.2f}\nTheo. Var: {theo_var:.2f}')

ax2.set_title('Stationary Distribution (Post Burn-in)', fontsize=14)
ax2.set_xlabel('State $Z$', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()