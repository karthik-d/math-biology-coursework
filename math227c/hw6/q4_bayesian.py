import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

# Data
y = np.array([120, 150, 100, 90, 160, 130, 110, 140])

# Prior Gamma(a, b), using shape a and rate b
a0 = 2.0
b0 = 0.4

# Posterior parameters for Poisson observations
# If y_i ~ Poisson(theta), then:
# theta | y ~ Gamma(a0 + sum(y_i), b0 + n)
n = len(y)
S = y.sum()

a_post = a0 + S
b_post = b0 + n

# Posterior mean and standard deviation
posterior_mean = a_post / b_post
posterior_sd = np.sqrt(a_post) / b_post

# Grid for plotting the posterior density
theta = np.linspace(max(1e-6, posterior_mean - 5 * posterior_sd),
                    posterior_mean + 5 * posterior_sd, 2000)

# Stable gamma density computation
log_pdf = (
    a_post * np.log(b_post)
    - gammaln(a_post)
    + (a_post - 1) * np.log(theta)
    - b_post * theta
)
posterior_pdf = np.exp(log_pdf - log_pdf.max())
posterior_pdf /= np.trapz(posterior_pdf, theta)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(theta, posterior_pdf, color='black', lw=2.5)
ax.axvline(
    posterior_mean,
    color='tab:blue',
    ls='--',
    lw=2,
    label=rf'Mean $\approx {posterior_mean:.2f}$'
)
ax.axvspan(
    posterior_mean - posterior_sd,
    posterior_mean + posterior_sd,
    color='tab:blue',
    alpha=0.12,
    label=rf'1 SD band $\approx [{posterior_mean-posterior_sd:.2f}, {posterior_mean+posterior_sd:.2f}]$'
)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel('Posterior density')
ax.set_title('Posterior for Poisson intensity with Gamma prior')
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

print(f"Posterior parameters: a = {a_post}, b = {b_post}")
print(f"Posterior mean: {posterior_mean:.4f}")
print(f"Posterior standard deviation: {posterior_sd:.4f}")