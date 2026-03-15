import numpy as np


def given_pde_system(N=101, L=1.0, D=0.01, c=0.1, v0=1.0):
    """
    Define the PDE system:
        du/dt = D d2u/dx2 + v(x) - c u
    on domain x in (0,1)
    Boundary conditions:
        du/dx = 0 at x=0 (Neumann)
        u = 0 at x=1 (Dirichlet)
    Initial condition: u(x,0) = 0
    Source: v(x) = v0 for x in (0,0.1), 0 elsewhere
    
    Returns:
        x: spatial grid
        u0: initial condition array
        dx: spatial step
        v: source term array
        D: diffusion coefficient
        c: decay coefficient
    """
    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    
    # Initial condition
    u0 = np.zeros(N)
    
    # Source term
    v = np.zeros_like(x)
    v[x < 0.1] = v0
    
    return x, u0, dx, v, D, c