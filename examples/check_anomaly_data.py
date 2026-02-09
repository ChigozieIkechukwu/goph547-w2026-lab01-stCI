# Script to check for and help with anomaly_data.mat
import os
from scipy.io import loadmat, savemat
import numpy as np

print("ANOMALY DATA CHECK")
print("="*60)

# Check for existing anomaly_data.mat
data_paths = [
    'anomaly_data.mat',
    'data/anomaly_data.mat',
    '../data/anomaly_data.mat',
    '../../data/anomaly_data.mat'
]

found = False
for path in data_paths:
    if os.path.exists(path):
        try:
            data = loadmat(path)
            print(f"✓ Found: {path}")
            print(f"  Arrays found: {[k for k in data.keys() if not k.startswith('__')]}")
            
            # Check if it has the required arrays
            required = ['x', 'y', 'z', 'rho']
            missing = [r for r in required if r not in data]
            
            if missing:
                print(f"✗ Missing arrays: {missing}")
            else:
                print(f"✓ All required arrays present")
                print(f"  x shape: {data['x'].shape}")
                print(f"  y shape: {data['y'].shape}")
                print(f"  z shape: {data['z'].shape}")
                print(f"  rho shape: {data['rho'].shape}")
                
                # Save a copy to data directory for consistency
                os.makedirs('data', exist_ok=True)
                savemat('data/anomaly_data.mat', data)
                print(f"✓ Saved copy to data/anomaly_data.mat")
            
            found = True
            break
            
        except Exception as e:
            print(f"✗ Error loading {path}: {e}")

if not found:
    print("✗ Could not find anomaly_data.mat")
    print("\nOPTIONS:")
    print("1. Place your anomaly_data.mat file in this directory")
    print("2. Or place it in the data/ subdirectory")
    print("3. Or create a test file using the code below")
    
    print("\nTo create a test anomaly for development:")
    print("""
    import numpy as np
    from scipy.io import savemat
    
    # Create a simple 3D grid
    x = np.arange(-50, 51, 2)
    y = np.arange(-50, 51, 2)
    z = np.arange(-100, -49, 2)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create a simple density anomaly
    center = [0, 0, -75]
    radius = 20
    distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
    rho = np.zeros_like(X)
    rho[distance < radius] = 1000
    rho[distance < radius/2] = 2000
    
    # Add noise
    np.random.seed(42)
    rho = rho + np.random.randn(*rho.shape) * 100
    rho = np.maximum(rho, 0)
    
    # Save
    data = {'x': X, 'y': Y, 'z': Z, 'rho': rho}
    savemat('data/anomaly_data.mat', data)
    print(f"Created test anomaly with shape: {rho.shape}")
    """)
