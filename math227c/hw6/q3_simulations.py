import numpy as np
import matplotlib.pyplot as plt

alpha = 0.08
beta = 1.0
T = 20.0
x0_frac = 0.5
ns = [10, 100]
npaths = 10

def gillespie_AB(alpha, beta, n, a0, T, rng):
    t = 0.0
    a = a0
    times = [t]
    avals = [a]

    while t < T:
        b = n - a

        # Reaction 1: A + B -> 2B, so A decreases by 1
        r1 = alpha * a * b

        # Reaction 2: B -> A, so A increases by 1
        r2 = beta * b

        rate = r1 + r2
        if rate <= 0:
            break

        dt = rng.exponential(1 / rate)
        t += dt
        if t > T:
            break

        if rng.random() < r1 / rate:
            a -= 1
        else:
            a += 1

        times.append(t)
        avals.append(a)

        if a == n:
            times.append(T)
            avals.append(a)
            break

    if times[-1] < T:
        times.append(T)
        avals.append(avals[-1])

    return np.array(times), np.array(avals)

def ode_solution(alpha, beta, n, x0, T, num=1000):
    # x(t) = A(t)/n
    # x' = -alpha*n*x*(1-x) + beta*(1-x)
    t = np.linspace(0, T, num)
    x = np.empty_like(t)
    x[0] = x0
    dt = t[1] - t[0]

    for i in range(1, num):
        f = lambda y: -alpha * n * y * (1 - y) + beta * (1 - y)

        k1 = f(x[i - 1])
        k2 = f(x[i - 1] + 0.5 * dt * k1)
        k3 = f(x[i - 1] + 0.5 * dt * k2)
        k4 = f(x[i - 1] + dt * k3)

        x[i] = x[i - 1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x[i] = min(max(x[i], 0.0), 1.0)

    return t, x

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

for ax, n in zip(axes, ns):
    a0 = int(n * x0_frac)
    colors = plt.cm.tab10(np.linspace(0, 1, npaths))

    for i in range(npaths):
        rng = np.random.default_rng(100 + i + n)
        t, a = gillespie_AB(alpha, beta, n, a0, T, rng)
        ax.step(t, a / n, where='post', color=colors[i], alpha=0.45, lw=1.4)

    t_ode, x_ode = ode_solution(alpha, beta, n, x0_frac, T)
    ax.plot(t_ode, x_ode, 'k--', lw=2.5, label='Deterministic ODE')

    ax.set_title(f'n = {n}')
    ax.set_ylabel(r'$A(t)/n$')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

axes[-1].set_xlabel('Time t')
fig.suptitle('Stochastic simulations versus deterministic ODE', y=0.98)
fig.tight_layout()
plt.show()