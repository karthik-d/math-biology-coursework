import numpy as np


def q1b_visual_check_f1():
	F = lambda v: np.array([np.log(v[0] + 2) + v[1]**2 - 3, v[0]*v[1] - 1])
	J = lambda v: np.array([[1/(v[0] + 2), 2*v[1]], [v[1], v[0]]])
	x0 = np.array([1.9, 1.85])  
	x = None
	return F, J, x0, x

def q1b_visual_check_f2():
	F = lambda v: np.array([v[0]**2 - 1, v[1]**2 - 4])
	J = lambda v: np.array([[2*v[0], 0], [0, 2*v[1]]])
	x0 = [0.2, 1]
	x = [1, 2]
	return F, J, x0, x

def q1c_f1():
    def F(v):
        x, y = v
        return np.array([
            x**2 + y - 2,
            x + np.sin(y) - 1
        ])
    def J(v):
        x, y = v
        return np.array([
            [2*x, 1],
            [1, np.cos(y)]
        ])
    x0 = [1.5, 0.5]
    x_true = None
    return F, J, x0, x_true


def q1c_f2():
    def F(v):
        x, y = v
        return np.array([
            np.exp(x) + y - 3,
            x + y**2 - 2
        ])
    def J(v):
        x, y = v
        return np.array([
            [np.exp(x), 1],
            [1, 2*y]
        ])
    x0 = [1.2, 0.8]
    x_true = None
    return F, J, x0, x_true


def q1c_f3():
    def F(v):
        x, y = v
        return np.array([
            np.exp(x) - y - 1,
            x**2 + y**2 - 3
        ])
    def J(v):
        x, y = v
        return np.array([
            [np.exp(x), -1],
            [2*x, 2*y]
        ])
    x0 = [1.5, 0.5]
    x_true = None
    return F, J, x0, x_true