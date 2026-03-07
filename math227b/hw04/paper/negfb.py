import numpy as np
import matplotlib.pyplot as plt
from pydelay import dde23

# =================================================================
# 1. PARAMETERS FOR FIGURE 1C (MCF7 CELL LINE)
# =================================================================
params = {
    'p0': 0.5, 'q0': 0.2,    # CSC division probabilities
    'p1': 0.5, 'q1': 0.1,    # PC division probabilities
    'v0': 1.0, 'v1': 2.0,    # Synthesis rates
    'd0': 0.01, 'd1': 0.05, 'd2': 0.1,  # Death rates
    'beta0': 2e-11,          # Delayed feedback strength
    'beta1': 3e-12,          # Instantaneous feedback strength
    'tau': 2.0               # Time delay (days)
}

# =================================================================
# 2. DEFINE THE DDE SYSTEM
# =================================================================
# x0 = CSCs, x1 = PCs, x2 = TDCs
# pydelay notation for delay: x2(t-tau)
eqns = {
    'x0': '(p0 - q0) * (v0 / (1 + beta0 * x2(t-tau)**2)) * x0 - d0 * x0',
    'x1': '(1 - p0 + q0) * (v0 / (1 + beta0 * x2(t-tau)**2)) * x0 + '
          '(p1 - q1) * (v1 / (1 + beta1 * x2**2)) * x1 - d1 * x1',
    'x2': '(1 - p1 + q1) * (v1 / (1 + beta1 * x2**2)) * x1 - d2 * x2'
}

# =================================================================
# 3. INITIALIZE AND SOLVE
# =================================================================
def run_simulation():
    # Initialize the dde23 solver
    dde = dde23(eqns=eqns, params=params)
    
    # Set initial conditions (Constant history for t < 0)
    # x0=10,000, x1=0, x2=0
    dde.set_p_init({'x0': 1e4, 'x1': 0.0, 'x2': 0.0})
    
    # Run simulation for 20 days
    # dtmax ensures the solver doesn't skip over the delay dynamics
    print("Integrating DDE using pydelay...")
    dde.run(tfinal=20, dtmax=0.01)
    
    # Sample the results at high resolution for plotting
    sol = dde.sample(0, 20, 0.05)
    return sol

# =================================================================
# 4. PLOTTING
# =================================================================
if __name__ == "__main__":
    sol = run_simulation()
    
    # Total population = x0 + x1 + x2
    total_population = sol['x0'] + sol['x1'] + sol['x2']
    
    plt.figure(figsize=(8, 6))
    plt.plot(sol['t'], total_population, color='blue', lw=2, label='Type I (pydelay)')
    
    # Formatting to match Figure 1 aesthetics from the paper
    plt.yscale('log')
    plt.ylim(1e4, 1e10)
    plt.xlim(0, 20)
    plt.xlabel('Culture Time (days)')
    plt.ylabel('Total Cell Number')
    plt.title('Reproduction of Figure 1C: Type I Feedback (pydelay)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.show()