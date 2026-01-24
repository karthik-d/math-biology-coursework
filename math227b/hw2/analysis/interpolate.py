import math
from pkg import lagrange


def run_interpolation(f, a, b, n, x_eval):
    """
    Generic interpolation driver.

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
    p = lagrange(x_nodes, f, x_eval)

    # true value.
    true_val = f(x_eval)

    return p, true_val


# Part 1(d): specific test functions. ----
def f1(x):
    return 3*x**3 + 4*x**2 + 2*x + 1

def f2(x):
    return math.sin(x)

def f3(x):
    return 1 / (1 + 25*x**2)


# ---- Run tests ----

if __name__ == "__main__":
    n = 6
    x_eval = 0.3

    p1, t1 = run_interpolation(f1, -1.0, 1.0, n, x_eval)
    print("f(x) = 3x^3 + 4x^2 + 2x + 1")
    print("Lagrange Interpolated:", p1)
    print("Exact:", t1)
    print()

    p2, t2 = run_interpolation(f2, 0.0, 2*math.pi, n, x_eval)
    print("f(x) = sin(x)")
    print("Lagrange Interpolated:", p2)
    print("Exact:", t2)
    print()

    p3, t3 = run_interpolation(f3, -1.0, 1.0, n, x_eval)
    print("f(x) = 1/(1+25x^2)")
    print("Lagrange Interpolated:", p3)
    print("Exact:", t3)