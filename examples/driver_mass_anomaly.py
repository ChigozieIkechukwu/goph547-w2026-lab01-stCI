# examples/driver_mass_anomaly.py
"""
Optimized distributed mass anomaly driver .

Key features:
- Uses the same physics convention as goph547lab01.gravity.gravity_effect_point:
  gz = G*m*(zobs - zs)/r^3 with G=6.674e-11
- Fast distributed forward modeling via chunked NumPy vectorization
- Cache/load survey data (anomaly_survey_data.mat)
- Optionally skip regenerating plots if they already exist
- Computes dg/dz and d2g/dz2 (Laplace relation)

Outputs (in examples/):
- anomaly_mean_density.png
- anomaly_survey_U.png
- anomaly_survey_gz.png
- anomaly_survey_derivatives.png
- anomaly_survey_data.mat
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goph547lab01.gravity import gravity_effect_point, gravity_potential_point  # noqa: E402


# -------------------------------------------------
# I/O
# -------------------------------------------------
def load_anomaly_data():
    """Load anomaly_data.mat from common locations."""
    possible_paths = [
        "anomaly_data.mat",
        "data/anomaly_data.mat",
        "../data/anomaly_data.mat",
        "../../data/anomaly_data.mat",
        "examples/anomaly_data.mat",
        "../examples/anomaly_data.mat",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            data = loadmat(path)
            print(f"✓ Loaded density data from: {path}")
            return data, os.path.abspath(path)
    raise FileNotFoundError("Could not find anomaly_data.mat in expected locations.")


def get_cache_path(script_dir):
    """Cache stored in examples/ (script directory)."""
    return os.path.join(script_dir, "anomaly_survey_data.mat")


# -------------------------------------------------
# Stats + Subregion
# -------------------------------------------------
def compute_basic_statistics(xm, ym, zm, rho, cell_size=2.0):
    """Compute total mass, barycenter, density stats; return stats dict and cell masses mm."""
    vcell = cell_size**3
    mm = rho * vcell
    mtot = float(np.sum(mm))

    xbar = float(np.sum(xm * mm) / mtot)
    ybar = float(np.sum(ym * mm) / mtot)
    zbar = float(np.sum(zm * mm) / mtot)

    stats = {
        "vcell": float(vcell),
        "mtot": mtot,
        "barycenter": np.array([xbar, ybar, zbar], dtype=float),
        "rho_max": float(np.max(rho)),
        "rho_mean": float(np.mean(rho)),
        "shape": rho.shape,
        "n_cells": int(np.prod(rho.shape)),
    }
    return stats, mm


def extract_subregion(mm, xm, ym, zm, rho, kx_min=40, kx_max=60, ky_min=44, ky_max=56, kz_min=7, kz_max=13):
    """
    Extract modeling subregion (index-based, Code-2 style).

    Assumes indexing is [ky, kx, kz]. If not, adapt slicing.
    Returns:
      flattened mass/coords for forward modeling, bounds, and rho_sub (3D) for Step-3 stats.
    """
    sl_y = slice(ky_min, ky_max + 1)
    sl_x = slice(kx_min, kx_max + 1)
    sl_z = slice(kz_min, kz_max + 1)

    mm_sub = mm[sl_y, sl_x, sl_z].reshape(-1)
    xm_sub = xm[sl_y, sl_x, sl_z].reshape(-1)
    ym_sub = ym[sl_y, sl_x, sl_z].reshape(-1)
    zm_sub = zm[sl_y, sl_x, sl_z].reshape(-1)
    rho_sub = rho[sl_y, sl_x, sl_z]

    bounds = dict(
        kx_min=kx_min,
        kx_max=kx_max,
        ky_min=ky_min,
        ky_max=ky_max,
        kz_min=kz_min,
        kz_max=kz_max,
    )
    return mm_sub, xm_sub, ym_sub, zm_sub, rho_sub, bounds


def report_subregion_statistics(xm, ym, zm, rho_sub, bounds, rho_global_mean):
    """Step-3 style report: coordinate ranges + mean density + comparison vs global mean."""
    ky_min, ky_max = bounds["ky_min"], bounds["ky_max"]
    kx_min, kx_max = bounds["kx_min"], bounds["kx_max"]
    kz_min, kz_max = bounds["kz_min"], bounds["kz_max"]

    # Coordinates in subregion (use the same slices)
    sl_y = slice(ky_min, ky_max + 1)
    sl_x = slice(kx_min, kx_max + 1)
    sl_z = slice(kz_min, kz_max + 1)

    x_sub = xm[sl_y, sl_x, sl_z]
    y_sub = ym[sl_y, sl_x, sl_z]
    z_sub = zm[sl_y, sl_x, sl_z]

    x_min, x_max = float(np.min(x_sub)), float(np.max(x_sub))
    y_min, y_max = float(np.min(y_sub)), float(np.max(y_sub))
    z_min, z_max = float(np.min(z_sub)), float(np.max(z_sub))

    rho_mean_region = float(np.mean(rho_sub))
    ratio = rho_mean_region / rho_global_mean if rho_global_mean != 0.0 else np.nan
    pct = (rho_mean_region - rho_global_mean) / rho_global_mean * 100.0 if rho_global_mean != 0.0 else np.nan

    print("\n" + "=" * 60)
    print("SUBREGION (Step 3) STATISTICS")
    print("=" * 60)
    print("Chosen as a non-negligible density region (index-defined subvolume).")
    print(f"X range: {x_min:.2f} to {x_max:.2f} m")
    print(f"Y range: {y_min:.2f} to {y_max:.2f} m")
    print(f"Z range: {z_min:.2f} to {z_max:.2f} m")
    print(f"Mean density in subregion: {rho_mean_region:.6f} kg/m³")
    print(f"Global mean density:        {rho_global_mean:.6f} kg/m³")
    print(f"Subregion / Global mean:    {ratio:.3f}×")
    print(f"Percent difference:         {pct:.2f} %")


# -------------------------------------------------
# Plot Mean Density + Subregion Box
# -------------------------------------------------
def plot_mean_density_with_box(xm, ym, zm, rho, stats, bounds, outpath):
    """3x1 mean density slices with barycenter marker (xk, markersize=3) and subregion box overlay."""
    xbar, ybar, zbar = stats["barycenter"]
    r_min, r_max = 0.0, 0.6

    kx_min, kx_max = bounds["kx_min"], bounds["kx_max"]
    ky_min, ky_max = bounds["ky_min"], bounds["ky_max"]
    kz_min, kz_max = bounds["kz_min"], bounds["kz_max"]

    plt.figure(figsize=(8, 9))

    def plot_slice(pos, X, Y, Zslice, xlabel, ylabel, title, x_mark, y_mark, box_coords, xlim, ylim):
        ax = plt.subplot(3, 1, pos)
        cf = ax.contourf(X, Y, Zslice, cmap="viridis_r", levels=np.linspace(r_min, r_max, 200))
        # SPEC-LITERAL MARKER: "xk" and markersize=3
        ax.plot(x_mark, y_mark, "xk", markersize=3)
        ax.plot(box_coords[0], box_coords[1], "--k")
        cbar = plt.colorbar(cf, ax=ax, ticks=np.linspace(r_min, r_max, 7))
        cbar.set_label(r"$\bar{\rho}$ [$kg/m^3$]")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # X-Z (mean along y)
    box_xz = (
        [xm[0, kx_min, 0], xm[0, kx_min, 0], xm[0, kx_max, 0], xm[0, kx_max, 0], xm[0, kx_min, 0]],
        [zm[0, 0, kz_min], zm[0, 0, kz_max], zm[0, 0, kz_max], zm[0, 0, kz_min], zm[0, 0, kz_min]],
    )
    plot_slice(
        1,
        xm[0, :, :],
        zm[0, :, :],
        np.mean(rho, axis=0),
        "x [m]",
        "z [m]",
        "Mean density along y-axis",
        xbar,
        zbar,
        box_xz,
        xlim=(-30, 30),
        ylim=(-20, 0),
    )

    # Y-Z (mean along x)
    box_yz = (
        [ym[ky_min, 0, 0], ym[ky_min, 0, 0], ym[ky_max, 0, 0], ym[ky_max, 0, 0], ym[ky_min, 0, 0]],
        [zm[0, 0, kz_min], zm[0, 0, kz_max], zm[0, 0, kz_max], zm[0, 0, kz_min], zm[0, 0, kz_min]],
    )
    plot_slice(
        2,
        ym[:, 0, :],
        zm[:, 0, :],
        np.mean(rho, axis=1),
        "y [m]",
        "z [m]",
        "Mean density along x-axis",
        ybar,
        zbar,
        box_yz,
        xlim=(-30, 30),
        ylim=(-20, 0),
    )

    # X-Y (mean along z)
    box_xy = (
        [xm[0, kx_min, 0], xm[0, kx_min, 0], xm[0, kx_max, 0], xm[0, kx_max, 0], xm[0, kx_min, 0]],
        [ym[ky_min, 0, 0], ym[ky_max, 0, 0], ym[ky_max, 0, 0], ym[ky_min, 0, 0], ym[ky_min, 0, 0]],
    )
    plot_slice(
        3,
        xm[:, :, 0],
        ym[:, :, 0],
        np.mean(rho, axis=2),
        "x [m]",
        "y [m]",
        "Mean density along z-axis",
        xbar,
        ybar,
        box_xy,
        xlim=(-30, 30),
        ylim=(-30, 30),
    )

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved mean density plot: {outpath}")


# -------------------------------------------------
# Fast Distributed Forward Modeling (Course-consistent physics)
# -------------------------------------------------
def forward_model_distributed_fast(X, Y, zp, mm_sub, xm_sub, ym_sub, zm_sub, chunk=20000, eps=0.0):
    """
    Exact summation over point masses accelerated via chunked vectorization.

    Physics matches goph547lab01.gravity:
      U  = G*m/r
      gz = G*m*(zobs - zs)/r^3
    """
    G = 6.674e-11  # match package default

    ny, nx = X.shape
    nz = len(zp)
    g_5 = np.zeros((ny, nx, nz), dtype=np.float64)
    U_5 = np.zeros((ny, nx, nz), dtype=np.float64)

    Xf = X.ravel()
    Yf = Y.ravel()
    ns = Xf.size

    # Filter out zero masses to save work
    keep = mm_sub != 0.0
    mm_sub = mm_sub[keep]
    xm_sub = xm_sub[keep]
    ym_sub = ym_sub[keep]
    zm_sub = zm_sub[keep]
    nm = mm_sub.size

    for k, zobs in enumerate(zp):
        gz_k = np.zeros(ns, dtype=np.float64)
        U_k = np.zeros(ns, dtype=np.float64)

        for s in range(0, nm, chunk):
            e = min(s + chunk, nm)

            m = mm_sub[s:e][:, None]
            xs = xm_sub[s:e][:, None]
            ys = ym_sub[s:e][:, None]
            zs = zm_sub[s:e][:, None]

            dx = Xf[None, :] - xs
            dy = Yf[None, :] - ys
            dz = (zobs - zs)

            r2 = dx * dx + dy * dy + dz * dz + eps
            r = np.sqrt(r2)

            inv_r = 1.0 / r
            inv_r3 = 1.0 / (r2 * r)

            U_k += np.sum(G * m * inv_r, axis=0)
            gz_k += np.sum(G * m * dz * inv_r3, axis=0)

        g_5[:, :, k] = gz_k.reshape(ny, nx)
        U_5[:, :, k] = U_k.reshape(ny, nx)

    return g_5, U_5


def generate_or_load_survey_fast(mm_sub, xm_sub, ym_sub, zm_sub, cache_file, chunk=20000, force_recompute=False):
    """Load cached survey if present; otherwise compute fast distributed survey and cache."""
    if (not force_recompute) and os.path.exists(cache_file):
        survey = loadmat(cache_file)
        x_5 = survey["x_5"]
        y_5 = survey["y_5"]
        zp = survey["zp"].flatten()
        g_5 = survey["g_5"]
        U_5 = survey["U_5"]
        print(f"✓ Loaded cached survey data: {cache_file}")
        return x_5, y_5, zp, g_5, U_5

    print("Survey cache not found or recompute forced. Running FAST distributed forward modeling...")

    x_5, y_5 = np.meshgrid(
        np.linspace(-100.0, 100.0, 41),
        np.linspace(-100.0, 100.0, 41),
    )
    zp = np.array([0.0, 1.0, 100.0, 110.0], dtype=float)

    g_5, U_5 = forward_model_distributed_fast(x_5, y_5, zp, mm_sub, xm_sub, ym_sub, zm_sub, chunk=chunk)

    savemat(cache_file, mdict={"x_5": x_5, "y_5": y_5, "zp": zp, "g_5": g_5, "U_5": U_5})
    print(f"✓ Survey computed and cached to: {cache_file}")
    return x_5, y_5, zp, g_5, U_5


# -------------------------------------------------
# Consistency spot-check (optional)
# -------------------------------------------------
def spot_check_against_partA_functions(x_5, y_5, zp, mm_sub, xm_sub, ym_sub, zm_sub, n_check=10, seed=0):
    """
    Demonstrate equivalence between the vectorized summation used here and the
    Part-A point formulas gravity_potential_point / gravity_effect_point.

    This is NOT used in production loops (kept fast); it checks a handful of points only.
    """
    rng = np.random.default_rng(seed)

    # Choose random station indices and random mass indices
    ny, nx = x_5.shape
    station_ids = rng.integers(0, ny * nx, size=n_check)
    mass_ids = rng.integers(0, mm_sub.size, size=n_check)
    z_ids = rng.integers(0, len(zp), size=n_check)

    # Single-point check: compare summing via Part A functions vs vectorized formula for that point
    max_abs_g = 0.0
    max_abs_U = 0.0

    for sid, zid in zip(station_ids, z_ids):
        iy = int(sid // nx)
        ix = int(sid % nx)
        obs = np.array([x_5[iy, ix], y_5[iy, ix], float(zp[zid])], dtype=float)

        # Part A style loop sum (only for a tiny sample)
        g_loop = 0.0
        U_loop = 0.0
        for m, xs, ys, zs in zip(mm_sub, xm_sub, ym_sub, zm_sub):
            src = np.array([xs, ys, zs], dtype=float)
            U_loop += gravity_potential_point(obs, src, float(m))
            g_loop += gravity_effect_point(obs, src, float(m))

        # Vectorized single-point (same algebra used in forward model)
        dx = obs[0] - xm_sub
        dy = obs[1] - ym_sub
        dz = obs[2] - zm_sub
        r2 = dx * dx + dy * dy + dz * dz
        r = np.sqrt(r2)
        U_vec = np.sum(6.674e-11 * mm_sub / r)
        g_vec = np.sum(6.674e-11 * mm_sub * dz / (r2 * r))

        max_abs_g = max(max_abs_g, float(abs(g_loop - g_vec)))
        max_abs_U = max(max_abs_U, float(abs(U_loop - U_vec)))

    print("\n" + "=" * 60)
    print("SPOT CHECK (equivalence to Part-A point functions)")
    print("=" * 60)
    print(f"Max |Δgz| over {n_check} random stations: {max_abs_g:.3e} m/s²")
    print(f"Max |ΔU|  over {n_check} random stations: {max_abs_U:.3e} m²/s²")


# -------------------------------------------------
# Derivatives
# -------------------------------------------------
def compute_derivatives(x_5, y_5, zp, g_5):
    """Compute dg/dz and d2g/dz2 using Laplace relation."""
    dx = float(x_5[0, 1] - x_5[0, 0])
    dy = float(y_5[1, 0] - y_5[0, 0])

    dgdz_0 = (g_5[:, :, 1] - g_5[:, :, 0]) / (zp[1] - zp[0])
    dgdz_100 = (g_5[:, :, 3] - g_5[:, :, 2]) / (zp[3] - zp[2])

    d2gdx2_0 = (g_5[1:-1, 2:, 0] - 2 * g_5[1:-1, 1:-1, 0] + g_5[1:-1, :-2, 0]) / dx**2
    d2gdy2_0 = (g_5[2:, 1:-1, 0] - 2 * g_5[1:-1, 1:-1, 0] + g_5[:-2, 1:-1, 0]) / dy**2
    d2gdz2_0 = -(d2gdx2_0 + d2gdy2_0)

    d2gdx2_100 = (g_5[1:-1, 2:, 2] - 2 * g_5[1:-1, 1:-1, 2] + g_5[1:-1, :-2, 2]) / dx**2
    d2gdy2_100 = (g_5[2:, 1:-1, 2] - 2 * g_5[1:-1, 1:-1, 2] + g_5[:-2, 1:-1, 2]) / dy**2
    d2gdz2_100 = -(d2gdx2_100 + d2gdy2_100)

    return dgdz_0, dgdz_100, d2gdz2_0, d2gdz2_100


# -------------------------------------------------
# Plotting
# -------------------------------------------------
def plot_gravity_maps(x_5, y_5, zp, g_5, outpath):
    """2x2 contour plots of gz at z = 0,1,100,110 m."""
    titles = [
        f"g_z at z = {zp[0]:.0f} m",
        f"g_z at z = {zp[1]:.0f} m",
        f"g_z at z = {zp[2]:.0f} m",
        f"g_z at z = {zp[3]:.0f} m",
    ]
    plt.figure(figsize=(10, 10))
    for idx in range(4):
        ax = plt.subplot(2, 2, idx + 1)
        cf = ax.contourf(x_5, y_5, g_5[:, :, idx], levels=50, cmap="plasma")
        plt.colorbar(cf, ax=ax, label="g_z [m/s²]")
        ax.set_title(titles[idx])
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved gravity maps: {outpath}")


def plot_potential_maps(x_5, y_5, zp, U_5, outpath):
    """2x2 contour plots of potential U at z = 0,1,100,110 m."""
    titles = [
        f"U at z = {zp[0]:.0f} m",
        f"U at z = {zp[1]:.0f} m",
        f"U at z = {zp[2]:.0f} m",
        f"U at z = {zp[3]:.0f} m",
    ]
    plt.figure(figsize=(10, 10))
    for idx in range(4):
        ax = plt.subplot(2, 2, idx + 1)
        cf = ax.contourf(x_5, y_5, U_5[:, :, idx], levels=50, cmap="viridis_r")
        plt.colorbar(cf, ax=ax, label="U [m²/s²]")
        ax.set_title(titles[idx])
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved potential maps: {outpath}")


def plot_derivative_maps(x_5, y_5, dgdz_0, dgdz_100, d2gdz2_0, d2gdz2_100, outpath):
    """2x2 derivative plots."""
    plt.figure(figsize=(10, 10))

    def levels_from(Z, n=50):
        lo, hi = np.percentile(Z, 2), np.percentile(Z, 98)
        if lo == hi:
            lo -= 1.0
            hi += 1.0
        return np.linspace(lo, hi, n)

    def quick_plot(pos, X, Y, Z, levels, label, title_text):
        ax = plt.subplot(2, 2, pos)
        cf = ax.contourf(X, Y, Z, cmap="viridis", levels=levels)
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(label)
        ax.text(-90, 70, title_text, weight="bold", bbox=dict(facecolor="white"))
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    quick_plot(1, x_5, y_5, dgdz_0, levels_from(dgdz_0), r"$\partial g_z/\partial z$ [s$^{-2}$]", "z = 0 m")
    quick_plot(3, x_5, y_5, dgdz_100, levels_from(dgdz_100), r"$\partial g_z/\partial z$ [s$^{-2}$]", "z = 100 m")
    quick_plot(
        2,
        x_5[1:-1, 1:-1],
        y_5[1:-1, 1:-1],
        d2gdz2_0,
        levels_from(d2gdz2_0),
        r"$\partial^2 g_z/\partial z^2$ [m$^{-1}$ s$^{-2}$]",
        "z = 0 m",
    )
    quick_plot(
        4,
        x_5[1:-1, 1:-1],
        y_5[1:-1, 1:-1],
        d2gdz2_100,
        levels_from(d2gdz2_100),
        r"$\partial^2 g_z/\partial z^2$ [m$^{-1}$ s$^{-2}$]",
        "z = 100 m",
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved derivative maps: {outpath}")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    # ---- user-tunable controls ----
    FORCE_RECOMPUTE_SURVEY = False   # True: ignore cache and recompute survey
    FORCE_REMAKE_PLOTS = False       # True: overwrite PNGs even if they exist
    CHUNK = 20000                    # raise if you have RAM (e.g., 50000); lower if memory issues
    RUN_SPOT_CHECK = True            # True: run a small equivalence check vs Part-A functions
    # -------------------------------

    print("=" * 70)
    print("GOPH 547 LAB 1 - PART C: DISTRIBUTED MASS ANOMALY")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1) Load density model
    data, data_path = load_anomaly_data()
    rho, xm, ym, zm = data["rho"], data["x"], data["y"], data["z"]

    # 2) Stats
    stats, mm = compute_basic_statistics(xm, ym, zm, rho, cell_size=2.0)
    xbar, ybar, zbar = stats["barycenter"]

    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    print(f"Loaded from: {data_path}")
    print(f"Total mass: {stats['mtot']:.3e} kg")
    print(f"Barycenter: ({xbar:.2f}, {ybar:.2f}, {zbar:.2f}) m")
    print(f"Max density: {stats['rho_max']:.3f} kg/m³")
    print(f"Mean density: {stats['rho_mean']:.6f} kg/m³")
    print(f"Grid shape: {stats['shape']} (N={stats['n_cells']:,})")

    # 3) Extract modeling subregion + Step-3 stats
    mm_sub, xm_sub, ym_sub, zm_sub, rho_sub, bounds = extract_subregion(mm, xm, ym, zm, rho)
    print("\n" + "=" * 60)
    print("MODELING SUBREGION")
    print("=" * 60)
    print(f"Subregion mass points: {mm_sub.size:,}")
    print(f"Index bounds: {bounds}")

    report_subregion_statistics(xm, ym, zm, rho_sub, bounds, rho_global_mean=stats["rho_mean"])

    # Output paths (in examples/)
    mean_density_out = os.path.join(script_dir, "anomaly_mean_density.png")
    U_out = os.path.join(script_dir, "anomaly_survey_U.png")
    gz_out = os.path.join(script_dir, "anomaly_survey_gz.png")
    deriv_out = os.path.join(script_dir, "anomaly_survey_derivatives.png")
    cache_file = get_cache_path(script_dir)

    # 4) Mean density plot (optional skip)
    if FORCE_REMAKE_PLOTS or (not os.path.exists(mean_density_out)):
        plot_mean_density_with_box(xm, ym, zm, rho, stats, bounds, mean_density_out)
    else:
        print(f"✓ Mean density plot exists: {mean_density_out}")

    # 5) Survey (cache)
    x_5, y_5, zp, g_5, U_5 = generate_or_load_survey_fast(
        mm_sub, xm_sub, ym_sub, zm_sub, cache_file, chunk=CHUNK, force_recompute=FORCE_RECOMPUTE_SURVEY
    )

    # Optional equivalence proof (tiny sample)
    if RUN_SPOT_CHECK:
        spot_check_against_partA_functions(x_5, y_5, zp, mm_sub, xm_sub, ym_sub, zm_sub, n_check=5, seed=0)

    # 6) Potential maps (optional skip)
    if FORCE_REMAKE_PLOTS or (not os.path.exists(U_out)):
        plot_potential_maps(x_5, y_5, zp, U_5, U_out)
    else:
        print(f"✓ Potential maps exist: {U_out}")

    # 7) Gravity maps (optional skip)
    if FORCE_REMAKE_PLOTS or (not os.path.exists(gz_out)):
        plot_gravity_maps(x_5, y_5, zp, g_5, gz_out)
    else:
        print(f"✓ Gravity maps exist: {gz_out}")

    # 8) Derivatives + plot (optional skip)
    if FORCE_REMAKE_PLOTS or (not os.path.exists(deriv_out)):
        dgdz_0, dgdz_100, d2gdz2_0, d2gdz2_100 = compute_derivatives(x_5, y_5, zp, g_5)
        plot_derivative_maps(x_5, y_5, dgdz_0, dgdz_100, d2gdz2_0, d2gdz2_100, deriv_out)
    else:
        print(f"✓ Derivative maps exist: {deriv_out}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print("Generated outputs:")
    print(f"  - {mean_density_out}")
    print(f"  - {U_out}")
    print(f"  - {gz_out}")
    print(f"  - {deriv_out}")
    print("Cached survey data:")
    print(f"  - {cache_file}")


if __name__ == "__main__":
    main()