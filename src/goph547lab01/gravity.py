import numpy as np

def gravity_potential_point(x, xm, m, G=6.674e-11):
    """Compute the gravity potential due to a point mass.
    
    Parameters
    ----------
    x : array_like, shape=(3,)
        Coordinates of survey point.
    xm : array_like, shape=(3,)
        Coordinates of point mass anomaly.
    m : float
        Mass of the anomaly.
    G : float, optional, default=6.674e-11
        Constant of gravitation. Default in SI units.
    
    Returns
    -------
    float
        Gravity potential at x due to anomaly at xm.
    """
    x = np.array(x)
    xm = np.array(xm)
    r = np.linalg.norm(x - xm)
    return G * m / r

def gravity_effect_point(x, xm, m, G=6.674e-11):
    """Compute the vertical gravity effect due to a point mass.
    
    Parameters
    ----------
    x : array_like, shape=(3,)
        Coordinates of survey point.
    xm : array_like, shape=(3,)
        Coordinates of point mass anomaly.
    m : float
        Mass of the anomaly.
    G : float, optional, default=6.674e-11
        Constant of gravitation. Default in SI units.
    
    Returns
    -------
    float
        Gravity effect at x due to anomaly at xm.
    """
    x = np.array(x)
    xm = np.array(xm)
    r_vec = x - xm
    r = np.linalg.norm(r_vec)
    
    # g_z = -∂U/∂z = G * m * (z - zm) / r^3
    return G * m * (x[2] - xm[2]) / (r**3)
