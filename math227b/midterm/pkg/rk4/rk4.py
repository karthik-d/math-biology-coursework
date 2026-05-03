import numpy as np

def rk4_solver(f, t0, y0, h, tf):
    """
    Classical 4th-order Runge-Kutta method with fixed final time.

    Parameters
    ----------
    f : callable
        RHS function f(t, y)
    t0 : float
        Initial time
    y0 : float or ndarray
        Initial value
    h : float
        Step size
    tf : float
        Final time

    Returns
    -------
    t : ndarray
        Time grid
    y : ndarray
        Numerical solution
    """

    # Number of steps (must be integer)
    N = int(np.round((tf - t0) / h))
    h = (tf - t0) / N   # enforce exact landing at tf

    t = t0 + np.arange(N+1) * h
    y = np.zeros((N+1,) + np.shape(y0), dtype=float)
    y[0] = y0

    for i in range(N):
        ti = t[i]
        wi = y[i]

        k1 = f(ti, wi)
        k2 = f(ti + 0.5*h, wi + 0.5*h*k1)
        k3 = f(ti + 0.5*h, wi + 0.5*h*k2)
        k4 = f(ti + h, wi + h*k3)

        y[i+1] = wi + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return t, y