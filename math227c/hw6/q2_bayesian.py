import numpy as np
import matplotlib.pyplot as plt

# Data
n = 100
k = 75

# Grid for theta
theta = np.linspace(1e-4, 1 - 1e-4, 1000)

# Prior: Uniform(0,1), so posterior is proportional to likelihood
# Posterior kernel: theta^k (1-theta)^(n-k)
posterior_unnorm = theta**k * (1 - theta)**(n - k)

# Normalize for plotting
posterior = posterior_unnorm / np.trapz(posterior_unnorm, theta)

# Posterior mean and 95% credible interval for Beta(k+1, n-k+1)
a_post = k + 1
b_post = n - k + 1
posterior_mean = a_post / (a_post + b_post)

try:
    from scipy.stats import beta
    ci_low, ci_high = beta.ppf([0.025, 0.975], a_post, b_post)
except Exception:
    ci_low, ci_high = 0.0, 1.0

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(theta, posterior, color="black", lw=2.5, label=r"Posterior density")
ax.axvline(posterior_mean, color="tab:blue", ls="--", lw=2,
           label=rf"Posterior mean $\approx {posterior_mean:.3f}$")
ax.axvspan(ci_low, ci_high, color="tab:blue", alpha=0.12,
           label=rf"Approx. 95% credible interval [{ci_low:.3f}, {ci_high:.3f}]")

ax.set_xlabel(r"$\theta$")
ax.set_ylabel("Posterior density up to a constant")
ax.set_title("Posterior distribution for the coin-flip problem")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
plt.show()