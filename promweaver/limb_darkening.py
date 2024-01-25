import numpy as np


def outgoing_chromo_ray_mu(mu, altitude_m):
    """
    Compute the outgoing mu (relative to the surface normal at the location of
    the ray hit). Assumes Sun is a sphere with Radius 695,700 km.
    Radius from: 2008ApJ...675L..53H

    Parameters
    ----------
    mu : float
        The mu of the ray of interest for the prominence.
    altitude_m : float
        The altitude of the prominence [m].

    Returns
    -------
    mu_out : float, optional
        The requested outgoing mu, or None if the ray does not hit the Sun.
    """
    Rm = 695_700_000
    outMu2 = 1.0 - (1.0 - mu**2) * (Rm + altitude_m) ** 2 / Rm**2
    if outMu2 < 0.0:
        # No hit.
        return None
    return np.sqrt(outMu2)
