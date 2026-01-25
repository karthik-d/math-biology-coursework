import math

from analysis.interpolate import run_lagrange, q1d_f1, q1d_f2, q1d_f3


if __name__ == "__main__":
    n = 500
    x_eval = 0.3

    p1, t1 = run_lagrange(q1d_f1, -1.0, 1.0, n, x_eval)
    print("f(x) = 3x^3 + 4x^2 + 2x + 1")
    print("Lagrange Interpolated:", p1)
    print("Exact:", t1)
    print()

    p2, t2 = run_lagrange(q1d_f2, 0.0, 2*math.pi, n, x_eval)
    print("f(x) = sin(x)")
    print("Lagrange Interpolated:", p2)
    print("Exact:", t2)
    print()

    p3, t3 = run_lagrange(q1d_f3, -1.0, 1.0, n, x_eval)
    print("f(x) = 1/(1+25x^2)")
    print("Lagrange Interpolated:", p3)
    print("Exact:", t3)