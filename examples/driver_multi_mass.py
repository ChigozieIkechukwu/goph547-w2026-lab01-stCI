import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from goph547lab01.gravity import gravity_potential_point, gravity_effect_point

def generate_mass_set(set_num):
    """Generate a set of 5 masses with same total mass and center of mass."""
    np.random.seed(set_num)  # For reproducibility
    
    # Target total mass and center of mass
    m_total = 1.0e7  # 10 million kg
    target_com = np.array([0, 0, -10])  # meters
    
    # Generate first 4 masses randomly
    masses = []
    positions = []
    
    for i in range(4):
        # Random mass with mean = m_total/5, std = m_total/100
        m_i = np.random.normal(m_total/5, m_total/100)
        masses.append(m_i)
        
        # Random position
        x_i = np.random.normal(0, 20)  # mean=0, std=20m
        y_i = np.random.normal(0, 20)  # mean=0, std=20m
        z_i = np.random.normal(-10, 2)  # mean=-10m, std=2m
        positions.append([x_i, y_i, z_i])
    
    masses = np.array(masses)
    positions = np.array(positions)
    
    # Calculate total mass and COM for first 4 masses
    m_4 = masses.sum()
    com_4 = np.sum(masses[:, np.newaxis] * positions, axis=0) / m_4
    
    # Calculate required 5th mass to satisfy constraints
    m5 = m_total - m_4
    if m5 <= 0:
        # Adjust if m5 would be negative
        scale_factor = m_total / m_4
        masses = masses * scale_factor
        m5 = m_total - masses.sum()
    
    # Calculate required position for 5th mass
    com5 = (m_total * target_com - m_4 * com_4) / m5
    
    # Add 5th mass
    masses = np.append(masses, m5)
    positions = np.vstack([positions, com5])
    
    # Verify constraints
    actual_total = masses.sum()
    actual_com = np.sum(masses[:, np.newaxis] * positions, axis=0) / actual_total
    
    print(f"Set {set_num}:")
    print(f"  Total mass: {actual_total:.2e} kg (target: {m_total:.2e})")
    print(f"  Center of mass: {actual_com} (target: {target_com})")
    print(f"  Masses: {masses}")
    
    return masses, positions

def create_multi_mass_plots(masses, positions, grid_spacing, set_num):
    """Create contour plots for multiple mass anomalies."""
    
    # Grid parameters
    x_min, x_max = -100, 100
    y_min, y_max = -100, 100
    
    # Create grid
    x_vals = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_vals = np.arange(y_min, y_max + grid_spacing, grid_spacing)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Elevations to plot
    elevations = [0, 10, 100]
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle(f'Mass Set {set_num} - Grid Spacing: {grid_spacing}m\nTotal Mass: 1.0e7 kg', fontsize=14)
    
    # First pass: find global ranges
    U_min, U_max = float('inf'), float('-inf')
    gz_min, gz_max = float('inf'), float('-inf')
    
    # Store grids for each elevation
    U_grids = []
    gz_grids = []
    
    for z in elevations:
        U_grid = np.zeros_like(X)
        gz_grid = np.zeros_like(X)
        
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                x = [X[j, k], Y[j, k], z]
                # Sum contributions from all masses
                U_total = 0
                gz_total = 0
                for m, pos in zip(masses, positions):
                    U_total += gravity_potential_point(x, pos, m)
                    gz_total += gravity_effect_point(x, pos, m)
                
                U_grid[j, k] = U_total
                gz_grid[j, k] = gz_total
        
        U_grids.append(U_grid)
        gz_grids.append(gz_grid)
        
        U_min = min(U_min, U_grid.min())
        U_max = max(U_max, U_grid.max())
        gz_min = min(gz_min, gz_grid.min())
        gz_max = max(gz_max, gz_grid.max())
    
    # Second pass: create plots
    for i, z in enumerate(elevations):
        # Plot gravity potential
        ax1 = axes[i, 0]
        contour1 = ax1.contourf(X, Y, U_grids[i], levels=50, cmap='viridis',
                                vmin=U_min, vmax=U_max)
        ax1.scatter(X, Y, color='black', marker='x', s=2, alpha=0.5)
        ax1.set_title(f'Potential at z = {z}m')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_aspect('equal')
        plt.colorbar(contour1, ax=ax1, label='Potential (m²/s²)')
        
        # Plot gravity effect
        ax2 = axes[i, 1]
        contour2 = ax2.contourf(X, Y, gz_grids[i], levels=50, cmap='plasma',
                                vmin=gz_min, vmax=gz_max)
        ax2.scatter(X, Y, color='black', marker='x', s=2, alpha=0.5)
        ax2.set_title(f'Gravity Effect at z = {z}m')
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_aspect('equal')
        plt.colorbar(contour2, ax=ax2, label='g_z (m/s²)')
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'mass_set_{set_num}_grid_{grid_spacing:.1f}m.png'
    plt.savefig(output_file, dpi=150)
    print(f"  Saved: {output_file}")
    
    return fig

if __name__ == "__main__":
    print("Generating multiple mass anomaly sets...")
    
    # Generate and save 3 sets of masses
    for set_num in range(1, 4):
        print(f"\n--- Generating Mass Set {set_num} ---")
        masses, positions = generate_mass_set(set_num)
        
        # Save to .mat file
        data = {
            'masses': masses,
            'positions': positions,
            'total_mass': masses.sum(),
            'center_of_mass': np.sum(masses[:, np.newaxis] * positions, axis=0) / masses.sum()
        }
        
        savemat(f'mass_set_{set_num}.mat', data)
        print(f"  Saved: mass_set_{set_num}.mat")
        
        # Create plots for both grid spacings
        create_multi_mass_plots(masses, positions, 5.0, set_num)
        create_multi_mass_plots(masses, positions, 25.0, set_num)
    
    print("\n✅ All mass sets generated and plotted!")
