import numpy as np


def f_linear(t, y, A):
    """Linear RHS: y' = A y."""
    return A @ y


def adams_bashforth_predictor_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args):
    """2-step AB predictor: y_{n+1}^* = y_n + h/2 (3 f_n - f_{n-1})."""
    f_n = f(t_n, y_n, *f_args)
    f_nm1 = f(t_nm1, y_nm1, *f_args)
    return y_n + 0.5 * h * (3.0 * f_n - f_nm1)


def adams_moulton_corrector_step(y_nm1, y_n, y_star, h, f, t_nm1, t_n, t_np1, *f_args):
    """1-step AM corrector: y_{n+1} = y_n + h/12 (-f_{n-1} + 5 f_n + 8 f^*)."""
    f_nm1 = f(t_nm1, y_nm1, *f_args)
    f_n = f(t_n, y_n, *f_args)
    f_star = f(t_np1, y_star, *f_args)
    return y_n + h / 12 * (-f_nm1 + 5 * f_n + 8 * f_star)


def predictor_corrector_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args):
    """Standard AB2-AM1 predictor-corrector step (both O(h^3) LTE)."""
    t_np1 = t_n + h
    y_star = adams_bashforth_predictor_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args)
    y_np1 = adams_moulton_corrector_step(y_nm1, y_n, y_star, h, f, t_nm1, t_n, t_np1, *f_args)
    return y_np1, y_star


def adams_bashforth_predictor_only_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args):
    """AB predictor-only as explicit 2nd-order method."""
    return adams_bashforth_predictor_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args)


def initialize_two_step_rk4(y0, t0, h, f, *f_args):
    """RK4 initialization for y1 (4th-order accurate startup)."""
    k1 = f(t0, y0, *f_args)
    k2 = f(t0 + 0.5*h, y0 + 0.5*h*k1, *f_args)
    k3 = f(t0 + 0.5*h, y0 + 0.5*h*k2, *f_args)
    k4 = f(t0 + h, y0 + h*k3, *f_args)
    y1 = y0 + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y0, y1, t0, t0 + h


def solve_adams_bashforth_predictor(f, t_span, y0, h, *f_args):
    """AB predictor-only solver (O(h^2) global)."""
    t0, T = t_span
    y0 = np.asarray(y0, dtype=float)
    m, N = y0.size, int(np.round((T - t0) / h))
    t = t0 + h * np.arange(N + 1)
    Y = np.zeros((N + 1, m)); Y[0] = y0

    y_nm1, y_n, t_nm1, t_n = initialize_two_step_rk4(y0, t0, h, f, *f_args)
    for n in range(N):
        y_np1 = adams_bashforth_predictor_only_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args)
        Y[n + 1] = y_np1
        y_nm1, y_n, t_nm1, t_n = y_n, y_np1, t_n, t[n + 1]
    return t, Y


def solve_predictor_corrector(f, t_span, y0, h, *f_args, store_predictor=False):
    """AB2-AM1 predictor-corrector solver (O(h^2) global, PC typically better)."""
    t0, T = t_span
    y0 = np.asarray(y0, dtype=float)
    m, N = y0.size, int(np.round((T - t0) / h))
    t = t0 + h * np.arange(N + 1)
    Y = np.zeros((N + 1, m)); Y[0] = y0
    Y_star = np.zeros_like(Y) if store_predictor else None

    y_nm1, y_n, t_nm1, t_n = initialize_two_step_rk4(y0, t0, h, f, *f_args)
    for n in range(N):
        y_np1, y_star = predictor_corrector_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args)
        Y[n + 1] = y_np1
        if store_predictor:
            Y_star[n + 1] = y_star
        y_nm1, y_n, t_nm1, t_n = y_n, y_np1, t_n, t[n + 1]
    return (t, Y, Y_star) if store_predictor else (t, Y)
