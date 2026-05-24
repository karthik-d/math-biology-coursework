import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def deterministic_predator_prey(Y, t, c1=1.0, c2=0.005, c3=0.6):
    """
    Defines the ODEs for the deterministic predator-prey model.
    
    Y: list or array containing [Y1, Y2]
    t: time variable
    """
    Y1, Y2 = Y
    
    # ODEs based on mass-action kinetics
    dY1_dt = c1 * Y1 - c2 * Y1 * Y2
    dY2_dt = c2 * Y1 * Y2 - c3 * Y2
    
    return [dY1_dt, dY2_dt]

# Set up initial conditions and time grid
Y0 = [50, 100]  # Initial prey (Y1) and predator (Y2)
t = np.linspace(0, 50.0, 1000)  # Time array for smooth plotting

# Solve the ODEs
solution = odeint(deterministic_predator_prey, Y0, t)
Y1_out = solution[:, 0]
Y2_out = solution[:, 1]

# ==========================================
# Plot 1: Time Course (Deterministic)
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(t, Y1_out, label='Prey ($Y_1$)', color='#1f77b4', linewidth=2)
plt.plot(t, Y2_out, label='Predator ($Y_2$)', color='#d62728', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Deterministic Predator-Prey Model: Time Course')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('time_course_det.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# Plot 2: Phase Plot (Deterministic)
# ==========================================
plt.figure(figsize=(8, 6))
plt.plot(Y1_out, Y2_out, color='purple', linewidth=2)
plt.xlabel('Prey Population ($Y_1$)')
plt.ylabel('Predator Population ($Y_2$)')
plt.title('Deterministic Predator-Prey Model: Phase Plot')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('phase_plot_det.png', dpi=300, bbox_inches='tight')
plt.show()