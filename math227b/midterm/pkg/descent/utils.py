import numpy as np


def check_hessian_pd(hess, x):
    H = hess(x)
    eigvals = np.linalg.eigvals(H)
    return H, eigvals
