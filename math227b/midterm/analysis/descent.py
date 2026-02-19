import numpy as np


def rosenbrock_functions():
    """
    Returns:
        f(x): scalar function
        grad(x): gradient (2-vector)
        hess(x): Hessian (2x2 matrix)
    """

    def f(x):
        x1, x2 = x
        return 100*(x2 - x1**2)**2 + (1 - x1)**2

    def grad(x):
        x1, x2 = x
        df_dx1 = -400*x1*(x2 - x1**2) - 2*(1 - x1)
        df_dx2 = 200*(x2 - x1**2)
        return np.array([df_dx1, df_dx2])

    def hess(x):
        x1, x2 = x
        h11 = 1200*x1**2 - 400*x2 + 2
        h12 = -400*x1
        h22 = 200
        return np.array([[h11, h12],
                         [h12, h22]])

    return f, grad, hess
