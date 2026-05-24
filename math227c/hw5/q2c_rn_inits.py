import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import odeint

# ==========================================
# 1. Deterministic Simulation
# ==========================================
def reversible_reaction_conc(y, t, k1, k2):
    a, b, c = y  # Concentrations
    net_rate = k1 * a * b - k2 * c
    return [-net_rate, -net_rate, net_rate]

# Base parameters
k1, k2 = 0.1, 0.05
y0_conc = [10.0, 5.0, 0.0]
T = 50.0
t_det = np.linspace(0, T, 500)

# The deterministic solution for concentrations remains the same across all scales
sol = odeint(reversible_reaction_conc, y0_conc, t_det, args=(k1, k2))
a_det, b_det, c_det = sol[:, 0], sol[:, 1], sol[:, 2]

# ==========================================
# 2. Stochastic Simulation Function
# ==========================================
def simulate_stochastic_scaled(V, k1_base=0.1, k2_base=0.05, A0_base=10, B0_base=5, C0_base=0, t_max=50.0):
    t = 0.0
    
    # Scale initial numbers by Volume factor V
    A = int(A0_base * V)
    B = int(B0_base * V)
    C = int(C0_base * V)
    
    # Scale bimolecular rate
    c1 = k1_base / V
    c2 = k2_base
    
    times = [t]
    A_vals, B_vals, C_vals = [A], [B], [C]
    
    while t < t_max:
        a1 = c1 * A * B
        a2 = c2 * C
        a0 = a1 + a2
        
        if a0 == 0:
            break
            
        r1, r2 = random.random(), random.random()
        tau = (1.0 / a0) * np.log(1.0 / r1)
        
        if t + tau > t_max:
            times.append(t_max)
            A_vals.append(A)
            B_vals.append(B)
            C_vals.append(C)
            break
            
        t += tau
        
        if r2 < a1 / a0:
            A -= 1; B -= 1; C += 1
        else:
            A += 1; B += 1; C -= 1
            
        times.append(t)
        A_vals.append(A)
        B_vals.append(B)
        C_vals.append(C)
        
    # Convert molecule numbers back to concentrations for comparison
    a_conc = np.array(A_vals) / V
    b_conc = np.array(B_vals) / V
    c_conc = np.array(C_vals) / V
    
    return times, a_conc, b_conc, c_conc

# ==========================================
# Plot: 2x3 Grid of Scaling Factors
# ==========================================
np.random.seed(42)
random.seed(42)

# Scaling factors (Volume multipliers)
V_factors = [1, 5, 10, 50, 100, 500]

fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, V in enumerate(V_factors):
    t_stoch, a_stoch, b_stoch, c_stoch = simulate_stochastic_scaled(V, t_max=T)
    
    ax = axes[i]
    
    # Plot Stochastic (Concentrations)
    ax.step(t_stoch, a_stoch, where='post', color='#1f77b4', alpha=0.6, linewidth=1.5)
    ax.step(t_stoch, b_stoch, where='post', color='#ff7f0e', alpha=0.6, linewidth=1.5)
    ax.step(t_stoch, c_stoch, where='post', color='#2ca02c', alpha=0.6, linewidth=1.5)
    
    # Plot Deterministic
    ax.plot(t_det, a_det, color='darkblue', linestyle='--', linewidth=1.5)
    ax.plot(t_det, b_det, color='darkorange', linestyle='--', linewidth=1.5)
    ax.plot(t_det, c_det, color='darkgreen', linestyle='--', linewidth=1.5)
    
    ax.set_title(f'Scaling Factor (V) = {V}\n($A_0={10*V}, B_0={5*V}$)')
    ax.grid(True, linestyle=':', alpha=0.7)
    
    if i >= 3:
        ax.set_xlabel('Time')
    if i % 3 == 0:
        ax.set_ylabel('Concentration')

# Add a single legend for the entire figure
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='#1f77b4', lw=2, alpha=0.6, label='[A] Stochastic'),
    Line2D([0], [0], color='darkblue', lw=2, linestyle='--', label='[A] Deterministic'),
    Line2D([0], [0], color='#ff7f0e', lw=2, alpha=0.6, label='[B] Stochastic'),
    Line2D([0], [0], color='darkorange', lw=2, linestyle='--', label='[B] Deterministic'),
    Line2D([0], [0], color='#2ca02c', lw=2, alpha=0.6, label='[C] Stochastic'),
    Line2D([0], [0], color='darkgreen', lw=2, linestyle='--', label='[C] Deterministic')
]
fig.legend(handles=custom_lines, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

plt.suptitle('Convergence of Stochastic to Deterministic Dynamics in the Thermodynamic Limit', fontsize=16)
plt.tight_layout()
plt.savefig('reversible_abc_convergence.png', dpi=300, bbox_inches='tight')
plt.show()