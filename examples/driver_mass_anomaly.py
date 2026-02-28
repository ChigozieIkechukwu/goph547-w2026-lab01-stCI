# examples/driver_mass_anomaly.py
"""
Important Tips:
- Same 5 point masses (mtot = 1.0e7 kg) at the same locations (z = -10 m)
- Same observation elevations: zp = [0.0, 10.0, 100.0]
- Same grids:
    * coarse: 9x9  (spacing 25 m over [-100, 100])
    * fine:   41x41 (spacing 5 m over [-100, 100])
- Same plotting layout: 3 rows x 2 cols (U left, g right), fixed color limits
- Uses the same Part-A physics functions:
    gravity_potential_point, gravity_effect_point
- Optional fast vectorized summation that is algebraically equivalent to the point functions
  (you can switch between "point" and "vectorized" implementations).

Outputs:
- examples/mass_anomaly_grid_25.png
- examples/mass_anomaly_grid_5.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from goph547lab01.gravity import (
    gravity_potential_point,
    gravity_effect_point,
)


# -------------------------------------------------
# Model setup (same as Code A)
# -------------------------------------------------
def define_mass_anomaly():
    mtot = 1.0e7
    m = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float) * mtot

    xm = np.array(
        [
            [-20.0, -20.0, -10.0],
            [20.0, -20.0, -10.0],
            [-20.0, 20.0, -10.0],
            [20.0, 20.0, -10.0],
            [0.0, 0.0, -10.0],
        ],
        dtype=float,
    )
    return m, xm


def make_xy_grid(npts):
    x = np.linspace(-100.0, 100.0, npts, dtype=float)
    return np.meshgrid(x, x)


# -------------------------------------------------
# Physics
# -------------------------------------------------
def evaluate_anomaly_point_loops(xg, yg, zp, m, xm):
    """
    Algorithmic structure (nested loops).
    Returns:
      U, g with shape (ny, nx, nz)
    """
    ny, nx = xg.shape
    nz = len(zp)

    U = np.zeros((ny, nx, nz), dtype=float)
    g = np.zeros((ny, nx, nz), dtype=float)

    xs = xg[0, :]
    ys = yg[:, 0]

    for k, z in enumerate(zp):
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                obs = [x, y, z]
                for mi, xsi in zip(m, xm):
                    U[iy, ix, k] += gravity_potential_point(obs, xsi, mi)
                    g[iy, ix, k] += gravity_effect_point(obs, xsi, mi)

    return U, g


def evaluate_anomaly_vectorized(xg, yg, zp, m, xm, eps=0.0):
    """
    Fast vectorized exact summation over point masses

    IMPORTANT:
    This assumes gravity_potential_point and gravity_effect_point implement the standard
    point-mass potential and vertical gravity component (gz), respectively:
      U  = G*m/r
      gz = G*m*(zobs - zs)/r^3
    If the gravity_effect_point returns something else, use evaluate_anomaly_point_loops.
    """
    # Try to infer G consistently:
    # We don't import G from the package, so we match the common value used in Code B.
    G = 6.674e-11

    ny, nx = xg.shape
    nz = len(zp)

    U = np.zeros((ny, nx, nz), dtype=float)
    g = np.zeros((ny, nx, nz), dtype=float)

    Xf = xg.ravel()[None, :]  # (1, ns)
    Yf = yg.ravel()[None, :]  # (1, ns)
    ns = Xf.shape[1]

    m = np.asarray(m, dtype=float)[:, None]    # (nm, 1)
    xs = np.asarray(xm[:, 0], dtype=float)[:, None]  # (nm, 1)
    ys = np.asarray(xm[:, 1], dtype=float)[:, None]
    zs = np.asarray(xm[:, 2], dtype=float)[:, None]

    dx = Xf - xs  # (nm, ns)
    dy = Yf - ys

    for k, zobs in enumerate(zp):
        dz = (float(zobs) - zs)  # (nm, 1)
        r2 = dx * dx + dy * dy + dz * dz + eps
        r = np.sqrt(r2)

        inv_r = 1.0 / r
        inv_r3 = 1.0 / (r2 * r)

        U_k = np.sum(G * m * inv_r, axis=0)          # (ns,)
        g_k = np.sum(G * m * dz * inv_r3, axis=0)    # (ns,)

        U[:, :, k] = U_k.reshape(ny, nx)
        g[:, :, k] = g_k.reshape(ny, nx)

    return U, g


# -------------------------------------------------
# Plotting 
# -------------------------------------------------
def plot_anomaly(x, y, U, g, spacing, outfile):
    fig = plt.figure(figsize=(8, 8))

    Umin, Umax = 0.0, 8.0e-5
    gmin, gmax = 0.0, 7.0e-6
    zlevels = ["0.0", "10.0", "100.0"]

    fig.suptitle(
        f"Mass anomaly, mtot = 1.0e7 kg, zm = -10 m, xy_grid = {spacing} m",
        weight="bold",
    )

    for k, ztxt in enumerate(zlevels):
        # Potential
        axU = plt.subplot(3, 2, 2 * k + 1)
        axU.contourf(
            x,
            y,
            U[:, :, k],
            levels=np.linspace(Umin, Umax, 500),
            cmap="viridis_r",
        )
        if spacing == 25.0:
            axU.plot(x, y, "xk", markersize=2)
        plt.colorbar(ticks=np.linspace(Umin, Umax, 5)).set_label(r"U [$m^2/s^2$]")
        axU.set_ylabel("y [m]")
        axU.text(
            -90,
            70,
            f"z = {ztxt} m",
            weight="bold",
            bbox=dict(facecolor="white"),
        )

        # Gravity
        axg = plt.subplot(3, 2, 2 * k + 2)
        axg.contourf(
            x,
            y,
            g[:, :, k],
            levels=np.linspace(gmin, gmax, 500),
            cmap="viridis_r",
        )
        if spacing == 25.0:
            axg.plot(x, y, "xk", markersize=2)
        plt.colorbar(ticks=np.linspace(gmin, gmax, 5)).set_label(r"g [$m/s^2$]")
        axg.text(
            -90,
            70,
            f"z = {ztxt} m",
            weight="bold",
            bbox=dict(facecolor="white"),
        )

    plt.xlabel("x [m]")

    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    plt.savefig(outfile, dpi=300)
    plt.close(fig)


# -------------------------------------------------
# Main driver 
# -------------------------------------------------
def main():
    # Choose computation method:
    # - "point": exactly matches Code A computation path (slow, but definitive)
    # - "vectorized": faster exact summation; assumes g is gz per standard formula
    METHOD = "point"  # change to "vectorized" if desired

    m, xm = define_mass_anomaly()
    zp = [0.0, 10.0, 100.0]

    # Coarse grid (9x9 => 25 m spacing)
    x25, y25 = make_xy_grid(9)
    if METHOD == "vectorized":
        U25, g25 = evaluate_anomaly_vectorized(x25, y25, zp, m, xm)
    else:
        U25, g25 = evaluate_anomaly_point_loops(x25, y25, zp, m, xm)

    plot_anomaly(
        x25,
        y25,
        U25,
        g25,
        spacing=25.0,
        outfile="examples/mass_anomaly_grid_25.png",
    )

    # Fine grid (41x41 => 5 m spacing)
    x5, y5 = make_xy_grid(41)
    if METHOD == "vectorized":
        U5, g5 = evaluate_anomaly_vectorized(x5, y5, zp, m, xm)
    else:
        U5, g5 = evaluate_anomaly_point_loops(x5, y5, zp, m, xm)

    plot_anomaly(
        x5,
        y5,
        U5,
        g5,
        spacing=5.0,
        outfile="examples/mass_anomaly_grid_5.png",
    )


if __name__ == "__main__":
    main()
