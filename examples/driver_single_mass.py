import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from goph547lab01.gravity import gravity_potential_point, gravity_effect_point


def create_contour_plots(grid_spacing: float) -> None:
    """Create contour plots for a single mass anomaly."""

    # Mass anomaly parameters
    m = 1.0e7  # 10 million kg
    xm = [0, 0, -10]  # centroid at (0, 0, -10 m)

    # Grid parameters
    x_min, x_max = -100, 100  # meters
    y_min, y_max = -100, 100  # meters

    # Create grid
    x_vals = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_vals = np.arange(y_min, y_max + grid_spacing, grid_spacing)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Elevations to plot
    elevations = [0, 10, 100]  # meters

    # Create figure with subplots
    # Use constrained_layout to prevent title/labels/colorbars from overlapping.
    fig, axes = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)

    # Use suptitle with padding so it doesn't collide with the top row.
    fig.suptitle(
        f'Gravity Potential and Effect for Single Mass Anomaly\nGrid Spacing: {grid_spacing} m',
        fontsize=14,
        y=1.02
    )

    # Global min/max for colorbars
    U_min, U_max = float('inf'), float('-inf')
    gz_min, gz_max = float('inf'), float('-inf')

    # First pass: compute values to find global ranges
    U_grids = []
    gz_grids = []

    for z in elevations:
        U_grid = np.zeros_like(X, dtype=float)
        gz_grid = np.zeros_like(X, dtype=float)

        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                x = [float(X[j, k]), float(Y[j, k]), float(z)]
                U_grid[j, k] = gravity_potential_point(x, xm, m)
                gz_grid[j, k] = gravity_effect_point(x, xm, m)

        U_grids.append(U_grid)
        gz_grids.append(gz_grid)

        U_min = min(U_min, float(U_grid.min()))
        U_max = max(U_max, float(U_grid.max()))
        gz_min = min(gz_min, float(gz_grid.min()))
        gz_max = max(gz_max, float(gz_grid.max()))

    # Second pass: create plots with consistent colorbars
    for i, z in enumerate(elevations):
        # Plot gravity potential
        ax1 = axes[i, 0]
        contour1 = ax1.contourf(
            X, Y, U_grids[i],
            levels=50,
            cmap='viridis',
            vmin=U_min, vmax=U_max
        )
        ax1.scatter(X, Y, color='black', marker='x', s=2, alpha=0.5)  # grid points
        ax1.set_title(f'Gravity Potential at z = {z} m')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_aspect('equal')

        # Use fig.colorbar (object-oriented) + pad spacing
        fig.colorbar(contour1, ax=ax1, label='Potential (m²/s²)', pad=0.02)

        # Plot gravity effect
        ax2 = axes[i, 1]
        contour2 = ax2.contourf(
            X, Y, gz_grids[i],
            levels=50,
            cmap='plasma',
            vmin=gz_min, vmax=gz_max
        )
        ax2.scatter(X, Y, color='black', marker='x', s=2, alpha=0.5)  # grid points
        ax2.set_title(f'Gravity Effect at z = {z} m')
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_aspect('equal')

        fig.colorbar(contour2, ax=ax2, label='g_z (m/s²)', pad=0.02)

    # Save the figure
    output_file = f'single_mass_grid_{grid_spacing:.1f}m.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_file}")

    plt.show()


if __name__ == "__main__":
    print("Creating contour plots for single mass anomaly...")

    # Create plots for both grid spacings
    create_contour_plots(5.0)
    create_contour_plots(25.0)

    print("Done!")

