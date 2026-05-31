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
npaths = 5

# Figure 1: stochastic sample paths
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

for ax, c2 in zip(axes, cases):
    for k in range(npaths):
        t, x = simulate_bdp(c1, c2, x0=x0, tmax=tmax, seed=100 + k)
        ax.step(t, x, where="post", linewidth=1.6, alpha=0.85)
    ax.set_title(rf"Sample paths for $c_1 = {c1}$, $c_2 = {c2}$")
    ax.set_ylabel(r"$X(t)$")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel(r"Time $t$")
fig.suptitle("Stochastic simulations of the birth-death process", y=0.98)
fig.tight_layout()
plt.show()

# Figure 2: deterministic mean-field model
t = np.linspace(0, tmax, 400)

fig2, ax2 = plt.subplots(figsize=(8, 4.8))
for c2 in cases:
    x_det = x0 * np.exp((c1 - c2) * t)
    ax2.plot(t, x_det, linewidth=2, label=rf"$c_2 = {c2}$")

ax2.set_title("Deterministic trajectories from the mean-field equation")
ax2.set_xlabel(r"Time $t$")
ax2.set_ylabel(r"$x(t)$")
ax2.grid(True, alpha=0.3)
ax2.legend()
fig2.tight_layout()
plt.show()