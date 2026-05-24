import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import odeint

# ==========================================
# 1. Deterministic Simulation (for comparison)
# ==========================================
def reversible_reaction(y, t, k1, k2):
    A, B, C = y
    net_rate = k1 * A * B - k2 * C
    return [-net_rate, -net_rate, net_rate]

k1, k2 = 0.1, 0.05
y0 = [10, 5, 0]
t_det = np.linspace(0, 50.0, 500)
sol = odeint(reversible_reaction, y0, t_det, args=(k1, k2))
A_det, B_det, C_det = sol[:, 0], sol[:, 1], sol[:, 2]

# ==========================================
# 2. Stochastic Simulation (Gillespie)
# ==========================================
def simulate_stochastic_abc(A0=10, B0=5, C0=0, k1=0.1, k2=0.05, t_max=50.0):
    t = 0.0
    A, B, C = A0, B0, C0
    
    times = [t]
    A_vals, B_vals, C_vals = [A], [B], [C]
    
    while t < t_max:
        # Propensities
        a1 = k1 * A * B  # Forward reaction
        a2 = k2 * C      # Backward reaction
        a0 = a1 + a2
        
        if a0 == 0:
            break
            
        r1, r2 = random.random(), random.random()
        tau = (1.0 / a0) * np.log(1.0 / r1)
        
        if t + tau > t_max:
            # Record final state at t_max and exit
            times.append(t_max)
            A_vals.append(A)
            B_vals.append(B)
            C_vals.append(C)
            break
            
        t += tau
        
        # Determine reaction
        if r2 < a1 / a0:
            A -= 1; B -= 1; C += 1  # A + B -> C
        else:
            A += 1; B += 1; C -= 1  # C -> A + B
            
        times.append(t)
        A_vals.append(A)
        B_vals.append(B)
        C_vals.append(C)
        
    return times, A_vals, B_vals, C_vals

# ==========================================
# Plot: 3 Stochastic Runs vs Deterministic
# ==========================================
plt.figure(figsize=(10, 6))

# Set seeds for reproducible simulation
np.random.seed(42)
random.seed(42)

# Run and plot the stochastic simulation 3 times
num_runs = 3
for i in range(num_runs):
    t_stoch, A_stoch, B_stoch, C_stoch = simulate_stochastic_abc()
    
    # Only label the first run to prevent duplicate legend entries
    label_A = '[A] Stochastic' if i == 0 else ""
    label_B = '[B] Stochastic' if i == 0 else ""
    label_C = '[C] Stochastic' if i == 0 else ""
    
    # Plot Stochastic (Step functions, using alpha=0.4 to see overlaps)
    plt.step(t_stoch, A_stoch, where='post', color='#1f77b4', alpha=0.4, linewidth=1.5, label=label_A)
    plt.step(t_stoch, B_stoch, where='post', color='#ff7f0e', alpha=0.4, linewidth=1.5, label=label_B)
    plt.step(t_stoch, C_stoch, where='post', color='#2ca02c', alpha=0.4, linewidth=1.5, label=label_C)

# Plot Deterministic (Dashed lines, thicker to stand out)
plt.plot(t_det, A_det, color='darkblue', linestyle='--', linewidth=2.5, label='[A] Deterministic')
plt.plot(t_det, B_det, color='darkorange', linestyle='--', linewidth=2.5, label='[B] Deterministic')
plt.plot(t_det, C_det, color='darkgreen', linestyle='--', linewidth=2.5, label='[C] Deterministic')

plt.xlabel('Time')
plt.ylabel('Number of Molecules / Concentration')
plt.title(r'Comparison of Stochastic (3 Runs) vs Deterministic Models: $A + B \rightleftharpoons C$')

# Adjust legend to be outside or organized so it doesn't block the data
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('reversible_abc_comparison_3runs.png', dpi=300, bbox_inches='tight')
plt.show()