import numpy as np
import matplotlib.pyplot as plt
import random

def simulate_sir_stochastic(N=100, I0=3, R0_target=0.5, gamma=0.1, t_max=150.0):
    """
    Simulates the stochastic SIR model using the Gillespie algorithm.
    """
    beta = R0_target * gamma / N
    
    t = 0.0
    S = N - I0
    I = I0
    R = 0
    
    times = [t]
    S_vals, I_vals, R_vals = [S], [I], [R]
    
    while t < t_max and I > 0:
        a1 = beta * S * I 
        a2 = gamma * I    
        a0 = a1 + a2
        
        if a0 == 0:
            break
            
        r1 = random.random()
        r2 = random.random()
        
        tau = (1.0 / a0) * np.log(1.0 / r1)
        t += tau
        
        if r2 < a1 / a0:
            S -= 1
            I += 1
        else:
            I -= 1
            R += 1
            
        times.append(t)
        S_vals.append(S)
        I_vals.append(I)
        R_vals.append(R)
        
    return times, S_vals, I_vals, R_vals

# Setup Parameters
N = 100
I0 = 3
gamma = 0.1 
R0_values = [0.9, 0.5, 0.2]
num_runs = 5
t_max = 100.0

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)

np.random.seed(42)
random.seed(42)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for idx, R0 in enumerate(R0_values):
    ax = axes[idx]
    
    for run in range(num_runs):
        t, S, I, R = simulate_sir_stochastic(N=N, I0=I0, R0_target=R0, gamma=gamma, t_max=t_max)
        
        # Plot only Infected curve. Added transparency for clarity.
        ax.step(t, I, where='post', color=colors[run], alpha=0.7, linewidth=1.5, label=f'Run {run+1}' if idx==0 else "")
        
    ax.set_title(f'$R_0 = {R0}$')
    ax.set_xlabel('Time')
    ax.grid(True, linestyle=':', alpha=0.5)
    
axes[0].set_ylabel('Infected Individuals ($I$)')
# Limit y-axis to a small number (e.g. 10) to clearly see the fade-out
axes[0].set_ylim(-0.5, 10) 

fig.legend(loc='center right', bbox_to_anchor=(1.08, 0.5))
plt.suptitle('Stochastic SIR Simulations: Disease Fade-out for $R_0 < 1$', fontsize=16)
plt.tight_layout()
plt.savefig('sir_stochastic_fadeout.png', dpi=300, bbox_inches='tight')
plt.show()