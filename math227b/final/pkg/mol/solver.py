# mol_solver.py
import numpy as np

def rhs(u, dx, D, c, v):
    N = len(u)
    dudt = np.zeros_like(u)
    
    # Internal points
    dudt[1:N-1] = D * (u[2:] - 2*u[1:N-1] + u[0:N-2]) / dx**2 + v[1:N-1] - c*u[1:N-1]
    
    # Boundary x=0 (Neumann: du/dx=0) => u[0] = u[1]
    dudt[0] = D * (u[1] - 2*u[0] + u[1]) / dx**2 + v[0] - c*u[0]
    
    # Boundary x=1 (Dirichlet: u=0)
    dudt[-1] = 0
    return dudt

def rk2_step(u, dt, dx, D, c, v):
    k1 = rhs(u, dx, D, c, v)
    k2 = rhs(u + dt * k1, dx, D, c, v)
    u_next = u + 0.5 * dt * (k1 + k2)
    u_next[-1] = 0  # enforce Dirichlet BC at x=1
    return u_next

def solve_pde_system(pde_func, dt=0.01, T=1.0, **kwargs):
    """
    Solve a PDE system defined by a callable function `pde_func`.
    
    pde_func: callable that returns (x, u0, dx, v, D, c)
    dt: time step
    T: final time
    kwargs: parameters passed to pde_func
    """
    x, u, dx, v, D, c = pde_func(**kwargs)
    
    t = 0.0
    usol = [u.copy()]
    times = [t]
    
    while t < T:
        u = rk2_step(u, dt, dx, D, c, v)
        t += dt
        usol.append(u.copy())
        times.append(t)
    
    return x, np.array(usol), np.array(times)