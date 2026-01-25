from typing import Callable, List, Union

def divided_difference(x: List[float], fvals: List[float]) -> List[float]:
    """
    Compute Newton divided differences for a set of nodes.

    Parameters
    ----------
    x : list of floats
        Interpolation nodes x0, x1, ..., xn
    fvals : list of floats
        Function values f(x0), f(x1), ..., f(xn)

    Returns
    -------
    a : list of floats
        Coefficients of the Newton polynomial:
        a[i] = f[x0, ..., xi]
    """
    n = len(x)
    table = [[0.0] * n for _ in range(n)]

    # first column is f(x_i)
    for i in range(n):
        table[i][0] = fvals[i]

    # fill divided difference table
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    # return first row (coefficients for Newton polynomial)
    return [table[0][j] for j in range(n)]


def compute_polynomial(a: List[float], x_nodes: List[float], x: float) -> float:
    """
    Evaluate Newton-form polynomial at a point using nested multiplication.

    Parameters
    ----------
    a : list of floats
        Newton coefficients a0, a1, ..., an
    x_nodes : list of floats
        Interpolation nodes x0, ..., x_{n-1} used for products
        Must have length len(a)-1
    x : float
        Point to evaluate polynomial at

    Returns
    -------
    float
        P(x)
    """
    n = len(a)
    value = a[-1]

    for k in range(n - 2, -1, -1):
        value = value * (x - x_nodes[k]) + a[k]
    return value


def interpolate(
    x_nodes: List[float],
    f: Union[Callable[[float], float], List[float]],
    x: float
) -> float:
    """
    Compute Lagrange interpolation at a point x.

    Uses Newton form: computes divided differences then evaluates
    the polynomial via nested multiplication.

    Parameters
    ----------
    x_nodes : list of floats
        Interpolation nodes
    f : callable or list of floats
        Function f(x) or precomputed values f(x_i)
    x : float
        Point to evaluate the interpolated polynomial

    Returns
    -------
    float
        Interpolated value P(x)
    """
    # Compute f(x_i) if f is callable
    if callable(f):
        fvals = [f(xi) for xi in x_nodes]
    else:
        fvals = f

    # Compute Newton coefficients
    a = divided_difference(x_nodes, fvals)
    return compute_polynomial(a, x_nodes[:len(a) - 1], x)
