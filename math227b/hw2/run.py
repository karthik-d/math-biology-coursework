import math
import autograd.numpy as anp

from analysis.lagrange import analyze_error


# ---- Q1(d) test functions ----
def q1d_f1(x):
    return 3*x**3 + 4*x**2 + 2*x + 1

def q1d_f2(x):
    return anp.sin(x)

def q1d_f3(x):
    return 1 / (1 + 25*x**2)


# ---- Run analysis for each function ----
if __name__ == "__main__":
    functions = [
        (q1d_f1, -1.0, 1.0, "3x^3 + 4x^2 + 2x + 1"),
        (q1d_f2, 0.0, 2*math.pi, "sin(x)"),
        (q1d_f3, -1.0, 1.0, "1/(1+25x^2)")
    ]

    for f, a, b, name in functions:
        analyze_error(f, a, b, name)