import math
from pkg import lagrange


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


# Part 1(d): specific test functions. ----
def q1d_f1(x):
    return 3*x**3 + 4*x**2 + 2*x + 1

def q1d_f2(x):
    return math.sin(x)

def q1d_f3(x):
    return 1 / (1 + 25*x**2)