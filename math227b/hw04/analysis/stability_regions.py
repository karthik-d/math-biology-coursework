import numpy as np
import matplotlib.pyplot as plt

# Grid in complex plane
x = np.linspace(-5, 2, 600)
y = np.linspace(-5, 5, 600)
X, Y = np.meshgrid(x, y)
Z = X + 1j*Y

rho = np.zeros_like(Z, dtype=float)

for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        z = Z[i, j]

        # coefficients of stability polynomial
        a = 1
        b = -(1 + 2*z + 0.75*z**2)
        c = 0.25*z**2

        roots = np.roots([a, b, c])

        rho[i, j] = np.max(np.abs(roots))

# Plot stability region
plt.figure(figsize=(7,6))
plt.contourf(X, Y, rho <= 1, levels=1)
plt.contour(X, Y, rho, levels=[1])
plt.axhline(0)
plt.axvline(0)

plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.title("Region of Absolute Stability")
plt.show()