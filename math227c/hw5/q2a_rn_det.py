import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def reversible_reaction(y, t, k1, k2):
    """
    Defines the ODEs for the reversible reaction A + B <-> C.
    
    y: list or array containing concentrations [A, B, C]
    t: time variable
    k1: forward reaction rate constant
    k2: backward reaction rate constant
    """
    A, B, C = y
    
    # Net rate of the forward reaction
    net_rate = k1 * A * B - k2 * C
    
    # ODEs based on mass-action kinetics
    dA_dt = -net_rate
    dB_dt = -net_rate
    dC_dt = net_rate
    
    return [dA_dt, dB_dt, dC_dt]

# Set parameters and initial conditions
k1 = 0.1   # Forward rate constant
k2 = 0.05  # Backward rate constant
y0 = [10.0, 5.0, 0.0]  # Initial concentrations of [A, B, C]

# Time grid
t = np.linspace(0, 50.0, 500)

# Solve the ODEs
solution = odeint(reversible_reaction, y0, t, args=(k1, k2))
A_out = solution[:, 0]
B_out = solution[:, 1]
C_out = solution[:, 2]

# ==========================================
# Plot: Deterministic Time Course
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(t, A_out, label='[A]', color='#1f77b4', linewidth=2.5)
plt.plot(t, B_out, label='[B]', color='#ff7f0e', linewidth=2.5)
plt.plot(t, C_out, label='[C]', color='#2ca02c', linewidth=2.5)

plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title(r'Deterministic Simulation of $A + B \rightleftharpoons C$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('reversible_abc_det.png', dpi=300, bbox_inches='tight')
plt.show()