import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

def gillespie_birth_death(lam, mu, initial_pop, max_time):
    """Simulate a continuous-time birth-death Markov process."""
    t = 0
    pop = initial_pop
    
    # Store trajectory
    times = [0]
    pops = [pop]
    
    while t < max_time:
        # Calculate rates for possible events
        r_birth = lam
        r_death = pop * mu
        r_total = r_birth + r_death
        
        if r_total == 0:
            break
            
        # Draw time to next event from an exponential distribution
        tau = np.random.exponential(1.0 / r_total)
        t += tau
        
        # Determine which event occurred proportionally to their rates
        if np.random.uniform(0, 1) < (r_birth / r_total):
            pop += 1  # Birth
        else:
            pop -= 1  # Death
            
        times.append(t)
        pops.append(pop)
        
    return np.array(times), np.array(pops)

# 1. Set Parameters
lam = 10.0
mu = 1.0
max_time = 1000.0  # Run long enough to establish stationary distribution
initial_pop = 0

# 2. Run simulation
np.random.seed(42)  # For reproducibility
times, pops = gillespie_birth_death(lam, mu, initial_pop, max_time)

# 3. Calculate empirical time-weighted distribution
max_pop = max(pops)
bins = np.arange(-0.5, max_pop + 1.5, 1)

# The time spent in state i is the difference between the time entering and leaving
time_spent = np.diff(times)
states = pops[:-1]

# Create a time-weighted histogram to get the empirical steady-state probabilities
hist, _ = np.histogram(states, bins=bins, weights=time_spent)
hist = hist / np.sum(hist) 

# 4. Calculate theoretical Poisson distribution
k_vals = np.arange(0, max_pop + 1)
theo_dist = poisson.pmf(k_vals, lam/mu)

# 5. Create figure with 2 subplots using Matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Birth-Death Process Simulation (λ=10, μ=1)', fontsize=16, y=1.02)

# Plot 1: Trajectory (only plot first 50 time units for clarity)
mask = times <= 50
ax1.step(times[mask], pops[mask], where='post', color='#1f77b4', linewidth=1.5, label='Population')
ax1.axhline(y=lam/mu, color='red', linestyle='--', linewidth=2, label='Expected Mean')

ax1.set_title('Population Trajectory (t=0 to 50)', fontsize=14)
ax1.set_xlabel('Time (t)', fontsize=12)
ax1.set_ylabel('Population Size', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=11)

# Plot 2: Distribution Comparison
ax2.bar(k_vals, hist, width=1, color='#1f77b4', edgecolor='white', alpha=0.6, label='Simulated Empirical')
ax2.plot(k_vals, theo_dist, color='red', marker='o', linestyle='-', linewidth=2, markersize=5, label='Theoretical Poisson')

ax2.set_title('Stationary Distribution vs Theoretical', fontsize=14)
ax2.set_xlabel('Population Size (k)', fontsize=12)
ax2.set_ylabel('Probability', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=11)

# Adjust layout to prevent overlap and display
plt.tight_layout()
plt.show()