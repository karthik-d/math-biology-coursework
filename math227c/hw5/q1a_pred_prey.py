import numpy as np
import matplotlib.pyplot as plt
import random

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

# Run the simulation
# Setting a seed ensures reproducibility for the assignment write-up
np.random.seed(42)
random.seed(42)
times, Y1_out, Y2_out = simulate_predator_prey(t_max=50.0)

# ==========================================
# Plot 1: Time Course
# ==========================================
plt.figure(figsize=(10, 5))
plt.step(times, Y1_out, label='Prey ($Y_1$)', color='#1f77b4', where='post', linewidth=1.5)
plt.step(times, Y2_out, label='Predator ($Y_2$)', color='#d62728', where='post', linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Stochastic Predator-Prey Model: Time Course')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('time_course.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# Plot 2: Phase Plot
# ==========================================
plt.figure(figsize=(8, 6))
plt.plot(Y1_out, Y2_out, color='purple', alpha=0.7, linewidth=1)
plt.xlabel('Prey Population ($Y_1$)')
plt.ylabel('Predator Population ($Y_2$)')
plt.title('Stochastic Predator-Prey Model: Phase Plot')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('phase_plot.png', dpi=300, bbox_inches='tight')
plt.show()