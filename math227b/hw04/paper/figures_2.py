import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- PARAMETERS (example values from Supplementary Table S2) ---
nu0_max = 0.5     # CSC division rate
nu1_max = 0.5     # PC division rate
d0 = 0.1          # CSC death rate
d1 = 0.1          # PC death rate
d2 = 0.5          # TDC death rate

p0_max = 0.5
q0_max = 0.2
p1_max = 0.5
q1_max = 0.2

beta0 = 5e-14
gamma1_0 = 6e-13
gamma2_0 = 2e-15
tau_delay = 2.0  # days

# Hill feedback functions
def hill(x, strength):
    return 1.0/(1.0 + strength * x**2)

# ODE system
def cancer_ode(y, t, hist):
    x0, x1, x2 = y

    # feedback using delayed TDC level
    # approximate delay by reading from history (simple implementation)
    # hist(t-tau) could be replaced by interpolation on stored past values
    delayed_x2 = hist(t - tau_delay) if t > tau_delay else x2

    p0 = p0_max * hill(delayed_x2, gamma1_0)
    q0 = q0_max * hill(delayed_x2, gamma2_0)
    nu0 = nu0_max * hill(delayed_x2, beta0)

    p1 = p1_max * hill(delayed_x2, gamma1_0)
    q1 = q1_max * hill(delayed_x2, gamma2_0)
    nu1 = nu1_max * hill(delayed_x2, beta0)

    # net self-renewal coefficients
    a0 = 2*p0 - 1
    a1 = 2*p1 - 1

    # compartment ODEs
    dx0dt = a0*nu0*x0 - d0*x0
    dx1dt = (2*q0*nu0*x0 + a1*nu1*x1) - d1*x1
    dx2dt = 2*q1*nu1*x1 - d2*x2

    return [dx0dt, dx1dt, dx2dt]

# Simple fixed history function for delay
def fixed_history(t):
    return 0.0

# initial condition
y0 = [1e4, 1e4, 1e4]  # initial populations

# time span
t = np.linspace(0, 20, 800)

# integrate
sol = odeint(cancer_ode, y0, t, args=(fixed_history,))

x0_t, x1_t, x2_t = sol.T

# Plot
plt.plot(t, x0_t, label="CSCs")
plt.plot(t, x1_t, label="PCs")
plt.plot(t, x2_t, label="TDCs")
plt.yscale('log')
plt.xlabel("Time (days)")
plt.ylabel("Cell Numbers")
plt.legend()
plt.show()