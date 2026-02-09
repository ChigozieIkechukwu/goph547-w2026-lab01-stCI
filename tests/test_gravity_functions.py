import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from goph547lab01.gravity import gravity_potential_point, gravity_effect_point

def test_gravity_potential_point():
    """Test gravity potential calculation for a point mass."""
    # Test case 1: Simple case
    x = [0, 0, 0]
    xm = [0, 0, -10]
    m = 1e7  # 10 million kg
    G = 6.674e-11
    
    U = gravity_potential_point(x, xm, m, G)
    
    # Expected: U = G*m/r = 6.674e-11 * 1e7 / 10 = 6.674e-5
    expected = G * m / 10
    assert abs(U - expected) < 1e-10, f"Expected {expected}, got {U}"
    print("✓ Test 1 passed: gravity_potential_point")
    
    # Test case 2: Different location
    x = [5, 5, 0]
    xm = [0, 0, -10]
    r = np.sqrt(5**2 + 5**2 + 10**2)  # sqrt(25 + 25 + 100) = sqrt(150)
    U = gravity_potential_point(x, xm, m, G)
    expected = G * m / r
    assert abs(U - expected) < 1e-10, f"Expected {expected}, got {U}"
    print("✓ Test 2 passed: gravity_potential_point")

def test_gravity_effect_point():
    """Test gravity effect calculation for a point mass."""
    # Test case 1: Directly above the mass
    x = [0, 0, 0]
    xm = [0, 0, -10]
    m = 1e7
    G = 6.674e-11
    
    gz = gravity_effect_point(x, xm, m, G)
    
    # Expected: gz = G * m * (z - zm) / r^3 = G * m * 10 / 1000 = G * m / 100
    expected = G * m * 10 / (10**3)
    assert abs(gz - expected) < 1e-10, f"Expected {expected}, got {gz}"
    print("✓ Test 1 passed: gravity_effect_point")
    
    # Test case 2: Off to the side
    x = [10, 0, 0]
    xm = [0, 0, -10]
    r = np.sqrt(10**2 + 0**2 + 10**2)  # sqrt(200)
    gz = gravity_effect_point(x, xm, m, G)
    expected = G * m * 10 / (r**3)
    assert abs(gz - expected) < 1e-10, f"Expected {expected}, got {gz}"
    print("✓ Test 2 passed: gravity_effect_point")

if __name__ == "__main__":
    test_gravity_potential_point()
    test_gravity_effect_point()
    print("\n✅ All tests passed!")
