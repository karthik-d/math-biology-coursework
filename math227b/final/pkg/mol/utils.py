import matplotlib.pyplot as plt
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


def plot_pde_solution(x, times, usol, overlay_times=None, overlay_x=None, cmap='viridis'):
    """
    Plot PDE solution in three subplots:
    1. Heatmap of u(x,t)
    2. Spatial profiles at selected times
    3. Temporal trajectories at selected spatial locations

    Parameters
    ----------
    x : array_like
        Spatial grid.
    times : array_like
        Time points.
    usol : 2D array
        Solution array of shape (nt, nx).
    overlay_times : list or array, optional
        Times at which to plot spatial profiles.
    overlay_x : list or array, optional
        Spatial locations at which to plot time trajectories.
    cmap : str
        Colormap for heatmap.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    # --------------------------------------
    # 1. Heatmap of u(x,t)
    # --------------------------------------
    ax = axes[0]
    extent = [x[0], x[-1], times[0], times[-1]]
    im = ax.imshow(usol, aspect='auto', origin='lower', extent=extent, cmap=cmap)
    fig.colorbar(im, ax=ax, label='u(x,t)')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('Heatmap of u(x,t)')

    # --------------------------------------
    # 2. Spatial profiles at selected times
    # --------------------------------------
    ax = axes[1]
    if overlay_times is not None:
        for t_overlay in overlay_times:
            idx = np.argmin(np.abs(times - t_overlay))
            ax.plot(x, usol[idx], label=f't={times[idx]:.2f}', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.set_title('Spatial profiles at select times')
    ax.grid(True)
    ax.legend()

    # --------------------------------------
    # 3. Temporal trajectories at selected spatial locations
    # --------------------------------------
    ax = axes[2]
    if overlay_x is not None:
        for x_val in overlay_x:
            idx = np.argmin(np.abs(x - x_val))
            ax.plot(times, usol[:, idx], label=f'x={x[idx]:.2f}', linewidth=2)
    ax.set_xlabel('t')
    ax.set_ylabel('u(t)')
    ax.set_title('Temporal trajectories at select x')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()