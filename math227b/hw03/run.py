import numpy as np

from pkg.newton.newton import newton_system
from pkg.newton.utils import plot_residuals


import numpy as np

def q1c_f1():
    # decoupled but nonlinear
    def F(v):
        x, y = v
        return np.array([
            np.sin(x),
            y**2 - 2
        ])
    def J(v):
        x, y = v
        return np.array([
            [np.cos(x), 0],
            [0, 2*y]
        ])
    x0 = [3.0, 0.3]          # farther away
    x_true = [0.0, np.sqrt(2)]
    return F, J, x0, x_true


def q1c_f2():
    # coupled, transcendental
    def F(v):
        x, y = v
        return np.array([
            x**2 - y - 1,
            x - np.cos(y)
        ])
    def J(v):
        x, y = v
        return np.array([
            [2*x, -1],
            [1, np.sin(y)]
        ])
    x0 = [2.5, 1.5]          # farther away
    x_true = None
    return F, J, x0, x_true


def q1c_f3():
    # Rosenbrock-type system (slow curvature)
    def F(v):
        x, y = v
        return np.array([
            10*(y - x**2),
            1 - x
        ])
    def J(v):
        x, y = v
        return np.array([
            [-20*x, 10],
            [-1, 0]
        ])
    x0 = [-1.5, 2.0]
    x_true = [1.0, 1.0]
    return F, J, x0, x_true


def q1c_f4():
    # exponential + polynomial coupling
    def F(v):
        x, y = v
        return np.array([
            np.exp(x) + y - 3,
            x**2 + y**2 - 4
        ])
    def J(v):
        x, y = v
        return np.array([
            [np.exp(x), 1],
            [2*x, 2*y]
        ])
    x0 = [1.5, 0.2]
    x_true = None
    return F, J, x0, x_true


def q1c_f5():
    # deliberately ill-scaled but smooth
    def F(v):
        x, y = v
        return np.array([
            x**3 - y,
            y**3 - x
        ])
    def J(v):
        x, y = v
        return np.array([
            [3*x**2, -1],
            [-1, 3*y**2]
        ])
    x0 = [0.8, 0.3]
    x_true = [0.0, 0.0]
    return F, J, x0, x_true


if __name__ == "__main__":

    tests = [q1c_f1, q1c_f2, q1c_f3, q1c_f4, q1c_f5]
    for i, testfun in enumerate(tests, 1):
        print(f"\nTEST FUNCTION {i}")
        F, J, x0, x_true = testfun()
        sol, info = newton_system(F, x0, J, x_true)
        print("solution:", sol)
        plot_residuals(info["error_history"])


