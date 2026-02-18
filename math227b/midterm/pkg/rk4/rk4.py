import numpy as np

def rk4_solver(f, t0, y0, h, tf):
    """
    Classical 4th-order Runge-Kutta method.

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
    t = np.arange(t0, tf + h, h)
    y = np.zeros_like(t)
    y[0] = y0

    for i in range(len(t) - 1):
        ti = t[i]
        wi = y[i]

        k1 = h * f(ti, wi)
        k2 = h * f(ti + 0.5*h, wi + 0.5*k1)
        k3 = h * f(ti + 0.5*h, wi + 0.5*k2)
        k4 = h * f(ti + h, wi + k3)

        y[i+1] = wi + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    return t, y
