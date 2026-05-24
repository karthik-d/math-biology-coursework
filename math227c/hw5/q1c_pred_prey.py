import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import odeint
from matplotlib.lines import Line2D

# ==========================================
# 1. Stochastic Simulation (Gillespie)
# ==========================================
def simulate_stochastic_low_pop(Y1_0=5, Y2_0=5, c1=1.0, c2=0.005, c3=0.6, t_max=15.0):
    t = 0.0
    Y1, Y2 = Y1_0, Y2_0
    
    times, Y1_vals, Y2_vals = [t], [Y1], [Y2]
    
    # Run while both populations exist and prey hasn't exploded
    while t < t_max and Y1 > 0 and Y2 > 0 and Y1 < 1000:
        a1 = c1 * Y1
        a2 = c2 * Y1 * Y2
        a3 = c3 * Y2
        a0 = a1 + a2 + a3
        
        if a0 == 0:
            break
            
        r1, r2 = random.random(), random.random()
        tau = (1.0 / a0) * np.log(1.0 / r1)
        t += tau
        
        if r2 < a1 / a0:
            Y1 += 1
        elif r2 < (a1 + a2) / a0:
            Y1 -= 1
            Y2 += 1
        else:
            Y2 -= 1
            
        times.append(t)
        Y1_vals.append(Y1)
        Y2_vals.append(Y2)
        
    # ABSORBING STATE 1: Predators go extinct. Prey grows exponentially.
    if Y2 == 0 and Y1 > 0 and t < t_max:
        while t < t_max and Y1 < 1000:
            a0 = c1 * Y1
            r1 = random.random()
            t += (1.0 / a0) * np.log(1.0 / r1)
            Y1 += 1
            times.append(t)
            Y1_vals.append(Y1)
            Y2_vals.append(0)
            
    # ABSORBING STATE 2: Prey goes extinct. Predators die off exponentially.
    elif Y1 == 0 and Y2 > 0 and t < t_max:
        while t < t_max and Y2 > 0:
            a0 = c3 * Y2
            r1 = random.random()
            t += (1.0 / a0) * np.log(1.0 / r1)
            Y2 -= 1
            times.append(t)
            Y1_vals.append(0)
            Y2_vals.append(Y2)
            
    return times, Y1_vals, Y2_vals

# Setting seeds for reproducibility so we get a mix of outcomes
np.random.seed(42)
random.seed(42)

# ==========================================
# Plot 1: Multiple Stochastic Runs
# ==========================================
num_simulations = 5
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#e377c2']

plt.figure(figsize=(12, 6))

for i in range(num_simulations):
    t_stoch, Y1_stoch, Y2_stoch = simulate_stochastic_low_pop(t_max=15.0)
    
    # Plot Prey (Solid line)
    plt.step(t_stoch, Y1_stoch, where='post', linestyle='-', color=colors[i], alpha=0.8, linewidth=1.5)
    # Plot Predator (Dashed line)
    plt.step(t_stoch, Y2_stoch, where='post', linestyle='--', color=colors[i], alpha=0.8, linewidth=1.5)

plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Stochastic Model: Multiple Realizations (Extinction & Survival Outcomes)')
plt.grid(True, linestyle=':', alpha=0.7)
# plt.ylim(-1, 50) 

# Create custom legend for clarity
custom_lines = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Prey ($Y_1$)'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='Predator ($Y_2$)')
]
for i in range(num_simulations):
    custom_lines.append(Line2D([0], [0], color=colors[i], lw=2, label=f'Run {i+1}'))

plt.legend(handles=custom_lines, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
plt.tight_layout()
plt.savefig('time_course_stoch_multiple.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 2. Deterministic Simulation (ODE)
# ==========================================
def deterministic_ode(Y, t, c1=1.0, c2=0.005, c3=0.6):
    Y1, Y2 = Y
    dY1_dt = c1 * Y1 - c2 * Y1 * Y2
    dY2_dt = c2 * Y1 * Y2 - c3 * Y2
    return [dY1_dt, dY2_dt]

t_det = np.linspace(0, 15.0, 1000)
solution = odeint(deterministic_ode, [5, 5], t_det)
Y1_det = solution[:, 0]
Y2_det = solution[:, 1]

# ==========================================
# Plot 2: Deterministic Time Course
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(t_det, Y1_det, label='Prey ($Y_1$)', color='black', linestyle='-', linewidth=2)
plt.plot(t_det, Y2_det, label='Predator ($Y_2$)', color='black', linestyle='--', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Deterministic Model: Continuous Recovery (No Extinction)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
# plt.ylim(-1, 50)
plt.savefig('time_course_det_recovery.png', dpi=300, bbox_inches='tight')
plt.show()