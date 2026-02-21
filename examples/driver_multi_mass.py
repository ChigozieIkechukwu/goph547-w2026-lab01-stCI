# examples/driver_multi_mass.py
"""
Lab 1B: Multiple Mass Anomalies (fully vectorized + spec-consistent)

Outputs (in examples/):
- mass_set_1.mat, mass_set_2.mat, mass_set_3.mat
- multi_mass_grid_25_set_1.png ... set_3.png
- multi_mass_grid_5_set_1.png  ... set_3.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat

# Add src directory to Python path (repo-friendly)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goph547lab01.gravity import gravity_potential_point, gravity_effect_point


# ----------------------------
# Config
# ----------------------------
MTOT = 1.0e7
TARGET_COM = np.array([0.0, 0.0, -10.0])

M_MEAN = MTOT / 5.0
M_STD = MTOT / 100.0
XSIG = np.array([20.0, 20.0, 2.0])

Z_MAX_ALLOWED = -1.0
ZP = np.array([0.0, 10.0, 100.0])

# Fixed color limits for fair comparison across sets/grids (as in the reference approach)
U_MIN, U_MAX = 0.0, 8.0e-5
G_MIN, G_MAX = 0.0, 7.0e-6

EXAMPLES_DIR = "examples"

# Physics constant: match the course package default (see goph547lab01.gravity defaults)
G_CONST = 6.674e-11


# ----------------------------
# Vectorized Physics (fast)
# ----------------------------
def gravity_potential_point_vec(obs_xyz: np.ndarray, src_xyz: np.ndarray, m: np.ndarray, G: float = G_CONST) -> np.ndarray:
    """
    Vectorized gravitational potential for multiple point masses.
    obs_xyz: (N,3), src_xyz: (M,3), m: (M,)
    returns: (N,)
    """
    r_vec = obs_xyz[None, :, :] - src_xyz[:, None, :]     # (M,N,3)
    r = np.linalg.norm(r_vec, axis=2)                     # (M,N)
    return np.sum(G * m[:, None] / r, axis=0)             # (N,)


def gravity_effect_point_vec(obs_xyz: np.ndarray, src_xyz: np.ndarray, m: np.ndarray, G: float = G_CONST) -> np.ndarray:
    """
    Vectorized vertical gravity effect gz for multiple point masses.
    Matches goph547lab01.gravity.gravity_effect_point convention:
      gz = G*m*(zobs - zs)/r^3
    obs_xyz: (N,3), src_xyz: (M,3), m: (M,)
    returns: (N,)
    """
    r_vec = obs_xyz[None, :, :] - src_xyz[:, None, :]     # (M,N,3)
    r2 = np.sum(r_vec * r_vec, axis=2)                    # (M,N)
    r = np.sqrt(r2)                                       # (M,N)
    dz = r_vec[:, :, 2]                                   # (M,N)
    return np.sum(G * m[:, None] * dz / (r2 * r), axis=0)  # (N,)


# ----------------------------
# Grid
# ----------------------------
def make_grid(npts: int):
    x = np.linspace(-100.0, 100.0, npts)
    return np.meshgrid(x, x)


# ----------------------------
# Mass Generator (Part B constraints)
# ----------------------------
def generate_mass_set(rng: np.random.Generator):
    """
    Generate one set of 5 masses such that:
      - sum(m) = MTOT
      - center of mass = TARGET_COM
      - all z <= Z_MAX_ALLOWED
    Strategy: sample first 4; solve for 5th mass and location; reject until valid.
    """
    while True:
        m = rng.normal(M_MEAN, M_STD, size=(5,))
        if np.any(m[:4] <= 0):
            continue

        xm = np.zeros((5, 3), dtype=float)
        xm[:4, 0] = rng.normal(TARGET_COM[0], XSIG[0], size=4)
        xm[:4, 1] = rng.normal(TARGET_COM[1], XSIG[1], size=4)
        xm[:4, 2] = rng.normal(TARGET_COM[2], XSIG[2], size=4)

        m5 = MTOT - float(np.sum(m[:4]))
        if m5 <= 0:
            continue
        m[4] = m5

        # Solve xm5 so that sum(m_i * x_i)/MTOT = TARGET_COM
        for i in range(3):
            xm[4, i] = (TARGET_COM[i] * MTOT - np.dot(m[:4], xm[:4, i])) / m[4]

        if not np.all(xm[:, 2] <= Z_MAX_ALLOWED):
            continue

        # Final verification
        com = (m @ xm) / np.sum(m)
        if np.allclose(np.sum(m), MTOT, rtol=0, atol=1e-6) and np.allclose(com, TARGET_COM, rtol=0, atol=1e-6):
            return m.astype(float), xm.astype(float)


def generate_mass_anomaly_sets(n_sets=3, seed=0):
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)

    for k in range(1, n_sets + 1):  # 1..3
        m, xm = generate_mass_set(rng)
        out = os.path.join(EXAMPLES_DIR, f"mass_set_{k}.mat")
        savemat(out, {"m": m.reshape(-1, 1), "xm": xm})

        print(f"✓ Saved {out}")
        print(f"  Verified Total Mass: {np.sum(m):.2e} kg")
        print(f"  Verified Center of Mass: {(m @ xm) / np.sum(m)}")
        print(f"  All masses below -1m: {np.all(xm[:, 2] <= -1.0)}")


# ----------------------------
# Vectorized Field Computation
# ----------------------------
def compute_gravity_fields(X, Y, zp, m, xm):
    """
    Compute U and gz on grid (X,Y) for elevations zp.
    Returns: U, g with shape (ny, nx, nz)
    """
    ny, nx = X.shape
    nz = len(zp)

    XY = np.column_stack([X.ravel(), Y.ravel()])
    N = XY.shape[0]

    U = np.zeros((N, nz), dtype=float)
    g = np.zeros((N, nz), dtype=float)

    for k, zobs in enumerate(zp):
        obs = np.column_stack([XY, np.full((N,), zobs, dtype=float)])
        U[:, k] = gravity_potential_point_vec(obs, xm, m)
        g[:, k] = gravity_effect_point_vec(obs, xm, m)

    return U.reshape(ny, nx, nz), g.reshape(ny, nx, nz)


# ----------------------------
# Course-function consistency spot-check
# ----------------------------
def spot_check_against_course_functions(X, Y, m, xm, tol_rel=1e-10, n_checks=8, seed=123):
    """
    Proves we are consistent with the course functions while staying vectorized.
    Checks a few random grid nodes and elevations.
    """
    rng = np.random.default_rng(seed)

    # Pick random indices in the grid
    ny, nx = X.shape
    ii = rng.integers(0, ny, size=n_checks)
    jj = rng.integers(0, nx, size=n_checks)
    kk = rng.integers(0, len(ZP), size=n_checks)

    for t in range(n_checks):
        i, j, k = int(ii[t]), int(jj[t]), int(kk[t])
        obs = np.array([float(X[i, j]), float(Y[i, j]), float(ZP[k])], dtype=float)

        # Course (scalar) sum
        U_s = 0.0
        g_s = 0.0
        for mi, xmi in zip(m, xm):
            U_s += float(gravity_potential_point(obs, xmi, float(mi)))
            g_s += float(gravity_effect_point(obs, xmi, float(mi)))

        # Vectorized at same single point
        obs1 = obs.reshape(1, 3)
        U_v = float(gravity_potential_point_vec(obs1, xm, m)[0])
        g_v = float(gravity_effect_point_vec(obs1, xm, m)[0])

        # Relative comparisons (avoid divide-by-zero)
        def rel_err(a, b):
            denom = max(1e-30, abs(a), abs(b))
            return abs(a - b) / denom

        eU = rel_err(U_s, U_v)
        eg = rel_err(g_s, g_v)

        if eU > tol_rel or eg > tol_rel:
            raise AssertionError(
                f"Spot-check failed at obs={obs}:\n"
                f"  U course={U_s:.6e}, U vec={U_v:.6e}, rel_err={eU:.3e}\n"
                f"  g course={g_s:.6e}, g vec={g_v:.6e}, rel_err={eg:.3e}"
            )

    print(f"✓ Spot-check passed ({n_checks} points): vectorized fields match course functions.")


# ----------------------------
# Plotting (spec-literal overlays + labels)
# ----------------------------
def plot_gravity_potential_and_effect(set_idx, X25, Y25, U25, g25, X5, Y5, U5, g5):
    def plot_block(X, Y, U, g, grid_spacing):
        fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
        fig.suptitle(
            f"Mass Set {set_idx} | mtot = 1.0e7 kg | Center of Mass z = -10 m\nGrid Spacing = {grid_spacing} m",
            weight="bold",
            fontsize=14,
        )

        for i, z_val in enumerate(ZP):
            # Potential
            axU = axes[i, 0]
            cfU = axU.contourf(
                X, Y, U[:, :, i],
                cmap="viridis_r",
                levels=np.linspace(U_MIN, U_MAX, 50),
                vmin=U_MIN, vmax=U_MAX,
            )
            fig.colorbar(cfU, ax=axU).set_label(r"U [$m^2/s^2$]")
            axU.set_title(f"Potential (U) at z = {z_val:.0f} m")

            # REQUIRED (Part A style): overlay grid points everywhere
            axU.plot(X, Y, "xk", markersize=2)

            # Axis labels on ALL plots
            axU.set_xlabel("x [m]")
            axU.set_ylabel("y [m]")
            axU.set_aspect("equal")

            # Gravity effect
            axg = axes[i, 1]
            cfg = axg.contourf(
                X, Y, g[:, :, i],
                cmap="magma",
                levels=np.linspace(G_MIN, G_MAX, 50),
                vmin=G_MIN, vmax=G_MAX,
            )
            fig.colorbar(cfg, ax=axg).set_label(r"$g_z$ [$m/s^2$]")
            axg.set_title(f"Gravity Effect ($g_z$) at z = {z_val:.0f} m")

            # REQUIRED overlay everywhere
            axg.plot(X, Y, "xk", markersize=2)

            # Axis labels on ALL plots
            axg.set_xlabel("x [m]")
            axg.set_ylabel("y [m]")
            axg.set_aspect("equal")

        return fig

    fig25 = plot_block(X25, Y25, U25, g25, 25.0)
    out25 = os.path.join(EXAMPLES_DIR, f"multi_mass_grid_25_set_{set_idx}.png")
    fig25.savefig(out25, dpi=300)
    plt.close(fig25)
    print(f"✓ Saved {out25}")

    fig5 = plot_block(X5, Y5, U5, g5, 5.0)
    out5 = os.path.join(EXAMPLES_DIR, f"multi_mass_grid_5_set_{set_idx}.png")
    fig5.savefig(out5, dpi=300)
    plt.close(fig5)
    print(f"✓ Saved {out5}")


# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(EXAMPLES_DIR, exist_ok=True)

    # Ensure the expected files exist (1..3). If not, generate.
    missing = [k for k in range(1, 4) if not os.path.exists(os.path.join(EXAMPLES_DIR, f"mass_set_{k}.mat"))]
    if missing:
        print("Generating mass sets...")
        generate_mass_anomaly_sets(n_sets=3, seed=0)

    # Grids: 25 m and 5 m
    X25, Y25 = make_grid(9)
    X5, Y5 = make_grid(41)

    for k in range(1, 4):
        print(f"\n--- Processing Mass Set {k} ---")
        data = loadmat(os.path.join(EXAMPLES_DIR, f"mass_set_{k}.mat"))
        m = data["m"][:, 0].astype(float)
        xm = data["xm"].astype(float)

        print(f"Verified Total Mass: {np.sum(m):.2e} kg")
        print(f"Verified Center of Mass: {(m @ xm) / np.sum(m)}")
        print(f"All masses below -1m: {np.all(xm[:, 2] <= -1.0)}")

        # Course-function consistency proof (fast, few points)
        spot_check_against_course_functions(X5, Y5, m, xm, tol_rel=1e-10, n_checks=8, seed=100 + k)

        # Vectorized compute
        U25, g25 = compute_gravity_fields(X25, Y25, ZP, m, xm)
        U5, g5 = compute_gravity_fields(X5, Y5, ZP, m, xm)

        # Plot + save
        plot_gravity_potential_and_effect(k, X25, Y25, U25, g25, X5, Y5, U5, g5)

    print("\n All mass sets processed and plotted!")


if __name__ == "__main__":
    main()