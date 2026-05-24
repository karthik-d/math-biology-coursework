import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.lines import Line2D


def simulate_predator_prey(Y1_0=50, Y2_0=100, c1=1.0, c2=0.005, c3=0.6, t_max=50.0):
    """
    Simulates a stochastic predator-prey model using the Gillespie Algorithm.
    
    Reactions:
    1) Y1 -> 2Y1          (Propensity: a1 = c1 * Y1)
    2) Y1 + Y2 -> 2Y2     (Propensity: a2 = c2 * Y1 * Y2)
    3) Y2 -> null         (Propensity: a3 = c3 * Y2)
    """
    
    # Initialize state variables
    t = 0.0
    Y1 = Y1_0
    Y2 = Y2_0
    
    # Lists to store the time course data
    times = [t]
    Y1_vals = [Y1]
    Y2_vals = [Y2]
    
    # Gillespie Algorithm loop
    while t < t_max and Y1 > 0 and Y2 > 0:
        # 1. Calculate propensities
        a1 = c1 * Y1
        a2 = c2 * Y1 * Y2
        a3 = c3 * Y2
        a0 = a1 + a2 + a3
        
        # Stop if no reactions can occur
        if a0 == 0:
            break
            
        # 2. Draw random numbers to determine time step and next reaction
        r1 = random.random()
        r2 = random.random()
        
        # Time to next reaction (exponentially distributed)
        tau = (1.0 / a0) * np.log(1.0 / r1)
        t += tau
        
        # Determine which reaction occurs
        # Reaction 1: Prey reproduction
        if r2 < a1 / a0:
            Y1 += 1
        # Reaction 2: Predation
        elif r2 < (a1 + a2) / a0:
            Y1 -= 1
            Y2 += 1
        # Reaction 3: Predator death
        else:
            Y2 -= 1
            
        # Record state
        times.append(t)
        Y1_vals.append(Y1)
        Y2_vals.append(Y2)
        
    return times, Y1_vals, Y2_vals


# Run the simulation 5 times
np.random.seed(42)
random.seed(42)

num_runs = 3
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
runs_data = []

for i in range(num_runs):
    times, Y1_out, Y2_out = simulate_predator_prey(t_max=50.0)
    runs_data.append((times, Y1_out, Y2_out))

# ==========================================
# Plot 1: Time Course (5 Runs)
# ==========================================
plt.figure(figsize=(12, 6))

for i, (times, Y1_out, Y2_out) in enumerate(runs_data):
    # Prey (Solid line)
    plt.step(times, Y1_out, where='post', linestyle='-', color=colors[i], alpha=0.7, linewidth=1.5)
    # Predator (Dashed line)
    plt.step(times, Y2_out, where='post', linestyle='--', color=colors[i], alpha=0.7, linewidth=1.5)

plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Stochastic Predator-Prey Model: Time Course (3 Runs)')

# Create a custom legend for clarity
custom_lines = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Prey ($Y_1$)'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='Predator ($Y_2$)')
]
for i in range(num_runs):
    custom_lines.append(Line2D([0], [0], color=colors[i], lw=2, label=f'Run {i+1}'))

plt.legend(handles=custom_lines, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('time_course.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# Plot 2: Phase Plot (5 Runs)
# ==========================================
plt.figure(figsize=(9, 7))

for i, (times, Y1_out, Y2_out) in enumerate(runs_data):
    # No step plot needed for phase plane, standard plot handles orbit trajectories well
    plt.plot(Y1_out, Y2_out, color=colors[i], alpha=0.6, linewidth=1.5, label=f'Run {i+1}')

plt.xlabel('Prey Population ($Y_1$)')
plt.ylabel('Predator Population ($Y_2$)')
plt.title('Stochastic Predator-Prey Model: Phase Plot (3 Runs)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('phase_plot.png', dpi=300, bbox_inches='tight')
plt.show()