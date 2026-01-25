import math
from pkg.lagrange import lagrange


def run_lagrange(f, a, b, n, x_eval):
    """
    generic interpolation driver for lagrange interpolation.

    Takes:
    f : callable
        Function to interpolate
    a, b : floats
        Interval endpoints
    n : int
        Number of interpolation points
    x_eval : float
        Point where polynomial is evaluated
    """
    
    # equally spaced nodes.
    x_nodes = [a + i * (b - a) / (n - 1) for i in range(n)]

    # interpolated value.
    p = lagrange.interpolate(x_nodes, f, x_eval)

    # true value.
    true_val = f(x_eval)

    return p, true_val
