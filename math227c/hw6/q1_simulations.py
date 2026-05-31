import numpy as np
import matplotlib.pyplot as plt

def simulate_bdp(c1, c2, x0=50, tmax=5.0, seed=0):
    rng = np.random.default_rng(seed)
    t = 0.0
    x = x0
    times = [t]
    xs = [x]

    while t < tmax and x > 0:
        birth = c1 * x
        death = c2 * x
        rate = birth + death
        if rate == 0:
            break

        dt = rng.exponential(1.0 / rate)
        t_next = t + dt
        if t_next > tmax:
            break

        if rng.random() < birth / rate:
            x += 1
        else:
            x -= 1

        t = t_next
        times.append(t)
        xs.append(x)

    return np.array(times), np.array(xs)

# Parameters
c1 = 10.0
cases = [9.5, 10.5]
x0 = 50
tmax = 10.0
npaths = 10

# Deterministic time grid
t_det = np.linspace(0, tmax, 400)

# One figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Use a colormap so the stochastic paths have different colors
colors = plt.cm.tab10(np.linspace(0, 1, npaths))

for ax, c2 in zip(axes, cases):
    # Plot stochastic trajectories
    for k in range(npaths):
        t, x = simulate_bdp(c1, c2, x0=x0, tmax=tmax, seed=100 + k)
        ax.step(
            t, x,
            where="post",
            color=colors[k],
            linewidth=1.6,
            alpha=0.55
        )

    # Overlay deterministic trajectory
    x_det = x0 * np.exp((c1 - c2) * t_det)
    ax.plot(
        t_det, x_det,
        linestyle="--",
        color="black",
        linewidth=2.2,
        label=r"Deterministic ODE"
    )

    ax.set_title(rf"$c_1 = {c1},\; c_2 = {c2}$")
    ax.set_ylabel(r"$X(t)$")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel(r"Time $t$")
fig.suptitle("Stochastic trajectories and deterministic solution", y=0.98)
fig.tight_layout()
plt.show()