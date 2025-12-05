# synthetic_cycle_demo.py
import numpy as np
import matplotlib.pyplot as plt
from math import pi

np.random.seed(1)

# Create circular manifold points
n = 400
radius = 1.0
theta = np.random.rand(n) * 2 * np.pi
theta[:100] = theta[:100] * 0.5
x = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)

# True tangential field and noisy RV
tangent = np.stack([-np.sin(theta), np.cos(theta)], axis=1)
tangent = tangent / (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12)
noise_scale = 0.4
rv_noisy = tangent + noise_scale * np.random.randn(*tangent.shape)
rv_noisy = rv_noisy / (np.linalg.norm(rv_noisy, axis=1, keepdims=True) + 1e-12)

# Partition into "early" (one semicircle) and "late" (opposite semicircle)
early_mask = (theta < 0.5*pi) | (theta > 1.5*pi)
late_mask = ~early_mask
idx_early = np.where(early_mask)[0]
idx_late = np.where(late_mask)[0]

# subsample for speed if needed
def subsample(idx, m=80):
    if len(idx) <= m: return idx
    return np.random.choice(idx, size=m, replace=False)
idx_early = subsample(idx_early, 80)
idx_late = subsample(idx_late, 80)

X_early = x[idx_early]
X_late = x[idx_late]
v_early = rv_noisy[idx_early]

# Cost matrices
def sq_euclidean(a,b):
    aa = np.sum(a*a, axis=1)[:,None]
    bb = np.sum(b*b, axis=1)[None,:]
    return aa + bb - 2*(a @ b.T)

M_euc = sq_euclidean(X_early, X_late)

# velocity alignment penalty (per pair)
dt = 1.0
lam = 5.0
displacements = X_late[None,:,:] - X_early[:,None,:]
percell_disp = displacements / dt
v_i = v_early[:,None,:]
M_vel = np.sum((percell_disp - v_i)**2, axis=2)

C_comb = M_euc + lam * M_vel

# Simple Sinkhorn (entropic OT).
def sinkhorn(a,b,C,reg, maxiter=500, tol=1e-6):
    K = np.exp(-C / reg)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(maxiter):
        u_prev = u.copy()
        u = a / (K @ v)
        v = b / (K.T @ u)
        if np.linalg.norm(u - u_prev, 1) < tol:
            break
    Gamma = np.diag(u) @ K @ np.diag(v)
    return Gamma

a = np.ones(len(X_early)) / len(X_early)
b = np.ones(len(X_late)) / len(X_late)
reg = 0.1

gamma_euc = sinkhorn(a,b,M_euc,reg)
mapped_euc = (gamma_euc @ X_late) / (gamma_euc.sum(axis=1)[:, None] + 1e-12)
disp_euc = mapped_euc - X_early

gamma_comb = sinkhorn(a,b,C_comb,reg)
mapped_comb = (gamma_comb @ X_late) / (gamma_comb.sum(axis=1)[:, None] + 1e-12)
disp_comb = mapped_comb - X_early

# Cosine metric with noisy RV
def cosine(a,b):
    num = np.sum(a*b, axis=1)
    den = np.linalg.norm(a,axis=1) * np.linalg.norm(b,axis=1) + 1e-12
    return num/den

cos_euc = cosine(disp_euc, v_early)
cos_comb = cosine(disp_comb, v_early)
print("Median cosine alignment (OT-only):", np.median(cos_euc))
print("Median cosine alignment (velocity-informed OT):", np.median(cos_comb))

# Plots (4 panels)
fig, axs = plt.subplots(2,2, figsize=(12,10))
axs = axs.flatten()
axs[0].scatter(x[:,0], x[:,1], s=6)
axs[0].quiver(x[::5,0], x[::5,1], tangent[::5,0], tangent[::5,1], angles='xy', scale_units='xy', scale=5)
axs[0].set_title("True tangential field (ground truth)"); axs[0].set_aspect('equal')

axs[1].scatter(x[:,0], x[:,1], s=6)
axs[1].quiver(x[::5,0], x[::5,1], rv_noisy[::5,0], rv_noisy[::5,1], angles='xy', scale_units='xy', scale=5)
axs[1].set_title("Noisy RNA-velocity (local)"); axs[1].set_aspect('equal')

axs[2].scatter(x[:,0], x[:,1], s=6)
axs[2].scatter(X_early[:,0], X_early[:,1], s=20)
axs[2].quiver(X_early[:,0], X_early[:,1], disp_euc[:,0], disp_euc[:,1], angles='xy', scale_units='xy', scale=0.5)
axs[2].set_title("OT-only (shortcuts)"); axs[2].set_aspect('equal')

axs[3].scatter(x[:,0], x[:,1], s=6)
axs[3].scatter(X_early[:,0], X_early[:,1], s=20)
axs[3].quiver(X_early[:,0], X_early[:,1], disp_comb[:,0], disp_comb[:,1], angles='xy', scale_units='xy', scale=0.5)
axs[3].set_title("Velocity-informed OT (follows loop)"); axs[3].set_aspect('equal')

plt.tight_layout()
plt.show()

# Histogram of cosine alignment
plt.figure(figsize=(6,4))
plt.hist(cos_euc, bins=30, alpha=0.6)
plt.hist(cos_comb, bins=30, alpha=0.6)
plt.legend(["OT-only", "velocity-informed OT"])
plt.title("Cosine alignment: mapped displacement vs RV (higher better)")
plt.show()
