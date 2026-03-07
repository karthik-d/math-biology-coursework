import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def get_base_params():
    """Base parameters from Table S2 best-fit for MCF7/neo cells."""
    return {
        'p0': 0.25, 'q0': 0.2,   # CSC division probabilities
        'p1': 0.3, 'q1': 0.1,   # PC division probabilities
        'v0': 1.0,              # CSC synthesis rate
        'v1': 2.0,              # PC synthesis rate (v0/v1 = 0.5)
        'd0': 0.01,             # CSC death rate (d0/d2 = 0.1)
        'd1': 0.05,             # PC death rate (d1/d2 = 0.5)
        'd2': 0.1               # TDC death rate
    }

# 1. Linear Model (No Feedback) - Eq. S1
def model_linear(y, t):
    """Simple cell lineage model without feedback control[cite: 1383]."""
    x0, x1, x2 = y
    p = get_base_params()
    
    dx0 = (p['p0'] - p['q0']) * p['v0'] * x0 - p['d0'] * x0
    dx1 = (1 - p['p0'] + p['q0']) * p['v0'] * x0 + (p['p1'] - p['q1']) * p['v1'] * x1 - p['d1'] * x1
    dx2 = (1 - p['p1'] + p['q1']) * p['v1'] * x1 - p['d2'] * x2
    return [dx0, dx1, dx2]

# 2. Type I Feedback Model (Rate Regulation) - Eq. S2
def model_type_I(y, t):
    """Feedback from TDCs regulating synthesis rates (v-parameters)[cite: 1397]."""
    x0, x1, x2 = y
    p = get_base_params()
    beta0 = 1.0e-13 # Strength of Type I feedback 
    beta1 = 1.0e-13
    
    # Negative feedback via Hill function with coefficient 2 [cite: 1396]
    f0 = 1 / (1 + beta0 * (x2*(t-p['tau_delay']))**2)
    f1 = 1 / (1 + beta1 * x2**2)
    
    dx0 = (p['p0'] - p['q0']) * (p['v0'] * f0) * x0 - p['d0'] * x0
    dx1 = (1 - p['p0'] + p['q0']) * (p['v0'] * f0) * x0 + (p['p1'] - p['q1']) * (p['v1'] * f1) * x1 - p['d1'] * x1
    dx2 = (1 - p['p1'] + p['q1']) * (p['v1'] * f1) * x1 - p['d2'] * x2
    return [dx0, dx1, dx2]

# 3. Type II Feedback Model (Division Probability) - Eq. S3
def model_type_II(y, t):
    """Feedback from TDCs regulating symmetric division probabilities (p, q)[cite: 1401]."""
    x0, x1, x2 = y
    p = get_base_params()
    gamma = 2.3e-17 # Strength of Type II feedback 
    
    # Negative feedback on division probabilities [cite: 1400]
    p0_eff = p['p0'] / (1 + gamma * x2**2)
    q0_eff = p['q0'] / (1 + gamma * x2**2)
    p1_eff = p['p1'] / (1 + gamma * x2**2)
    q1_eff = p['q1'] / (1 + gamma * x2**2)
    
    dx0 = (p0_eff - q0_eff) * p['v0'] * x0 - p['d0'] * x0
    dx1 = (1 - p0_eff + q0_eff) * p['v0'] * x0 + (p1_eff - q1_eff) * p['v1'] * x1 - p['d1'] * x1
    dx2 = (1 - p1_eff + q1_eff) * p['v1'] * x1 - p['d2'] * x2
    return [dx0, dx1, dx2]

# 4. Combined Type I & II Feedback Model - Eq. S4
def model_combined(y, t):
    """Combined negative feedbacks on both division rates and probabilities[cite: 1405]."""
    x0, x1, x2 = y
    p = get_base_params()
    beta = 1.0e-13 
    gamma = 2.3e-17
    
    f_rate = 1 / (1 + beta * x2**2)
    p0_eff = p['p0'] / (1 + gamma * x2**2)
    q0_eff = p['q0'] / (1 + gamma * x2**2)
    p1_eff = p['p1'] / (1 + gamma * x2**2)
    q1_eff = p['q1'] / (1 + gamma * x2**2)
    
    dx0 = (p0_eff - q0_eff) * (p['v0'] * f_rate) * x0 - p['d0'] * x0
    dx1 = (1 - p0_eff + q0_eff) * (p['v0'] * f_rate) * x0 + (p1_eff - q1_eff) * (p['v1'] * f_rate) * x1 - p['d1'] * x1
    dx2 = (1 - p1_eff + q1_eff) * (p['v1'] * f_rate) * x1 - p['d2'] * x2
    return [dx0, dx1, dx2]

# --- DRIVER CODE ---

def run_reproduction_fig1():
    # 1. Simulation parameters
    t = np.linspace(0, 20, 1000)  # Time in days
    y0 = [1.8e2, 0, 0]              # Initial conditions: 10,000 CSCs, 0 PCs, 0 TDCs
    
    # 2. Run simulations for each model
    # Note: These refer to the independent functions implemented previously
    sol_linear   = odeint(model_linear, y0, t)
    # sol_type_I   = odeint(model_type_I, y0, t)
    # sol_type_II  = odeint(model_type_II, y0, t)
    # sol_combined = odeint(model_combined, y0, t)
    
    # 3. Calculate Total Cell Number (x0 + x1 + x2) for each
    total_linear   = np.sum(sol_linear, axis=1)
    # total_type_I   = np.sum(sol_type_I, axis=1)
    # total_type_II  = np.sum(sol_type_II, axis=1)
    # total_combined = np.sum(sol_combined, axis=1)
    
    # 4. Plotting (Reproducing Figure 1 aesthetics)
    plt.figure(figsize=(8, 6))
    
    plt.plot(t, total_linear, 'k--', label='No Feedback (Linear)')
    # plt.plot(t, total_type_I, 'b-', label='Type I (Rate Control)')
    # plt.plot(t, total_type_II, 'g-', label='Type II (Division Control)')
    # plt.plot(t, total_combined, 'r-', linewidth=2, label='Combined (Type I + II)')
    
    # Standard figure formatting per the paper
    plt.ylim(0, 4e6)
    plt.xlim(0, 20)
    plt.xlabel('Culture Time (days)', fontsize=12)
    plt.ylabel('Cell Number ($log_{10}$)', fontsize=12)
    plt.title('Figure 1: Comparison of Growth Kinetics Models', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_reproduction_fig1()