import numpy as np


def f_linear(t, y, A):
    """Linear RHS: y' = A y."""
    return A @ y


def adams_bashforth_predictor_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args):
    """
    Predictor Step:
    y^*_{n+1} = y_n + (h/2) * (3*f(y_n) - f(y_{n-1}))
    """
    f_n = f(t_n, y_n, *f_args)
    f_nm1 = f(t_nm1, y_nm1, *f_args)
    return y_n + 0.5 * h * (3.0 * f_n - f_nm1)


def predictor_corrector_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args):
    """
    Full Predictor-Corrector Scheme:
    1. Predict y^*_{n+1} using AB2.
    2. Correct y_{n+1} using Trapezoidal: y_{n+1} = y_n + (h/2)*(f(y_n) + f(y^*_{n+1}))
    """
    t_np1 = t_n + h
    f_n = f(t_n, y_n, *f_args)
    
    # Predict
    y_star = adams_bashforth_predictor_step(y_nm1, y_n, h, f, t_nm1, t_n, *f_args)
    
    # Evaluate at predicted state (note: autonomous f ignores t_np1, but kept general)
    f_star = f(t_np1, y_star, *f_args)
    
    # Correct
    y_np1 = y_n + 0.5 * h * (f_n + f_star)
    return y_np1, y_star


def initialize_two_step_rk4(y0, t0, h, f, *f_args):
    """RK4 to get y1 from y0 (two-step startup)."""
    k1 = f(t0, y0, *f_args)
    k2 = f(t0 + 0.5*h, y0 + 0.5*h*k1, *f_args)
    k3 = f(t0 + 0.5*h, y0 + 0.5*h*k2, *f_args)
    k4 = f(t0 + h, y0 + h*k3, *f_args)
    return y0 + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def solve_adams_bashforth_predictor(f, t_span, y0, h, *f_args):
    """AB2 predictor only (explicit multistep)."""
    t0, T = t_span
    y0 = np.asarray(y0, dtype=float)
    N = int(np.round((T - t0) / h))
    t = t0 + h * np.arange(N + 1)
    Y = np.zeros((N + 1, y0.size))
    
    Y[0] = y0
    if N > 0:
        Y[1] = initialize_two_step_rk4(Y[0], t[0], h, f, *f_args)
    
    for n in range(1, N):
        Y[n+1] = adams_bashforth_predictor_step(Y[n-1], Y[n], h, f, t[n-1], t[n], *f_args)
    return t, Y


def solve_predictor_corrector(f, t_span, y0, h, *f_args, store_predictor=False):
    """AB2-Trapezoidal PECE predictor-corrector."""
    t0, T = t_span
    y0 = np.asarray(y0, dtype=float)
    N = int(np.round((T - t0) / h))
    t = t0 + h * np.arange(N + 1)
    Y = np.zeros((N + 1, y0.size))
    Y_star = np.zeros_like(Y) if store_predictor else None

    Y[0] = y0
    if N > 0:
        Y[1] = initialize_two_step_rk4(Y[0], t[0], h, f, *f_args)
    
    for n in range(1, N):
        y_np1, y_star = predictor_corrector_step(Y[n-1], Y[n], h, f, t[n-1], t[n], *f_args)
        Y[n+1] = y_np1
        if store_predictor:
            Y_star[n+1] = y_star
            
    return (t, Y, Y_star) if store_predictor else (t, Y)
