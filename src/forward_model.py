"""
Forward Modeling Module for Gravity Anomaly of a Buried Sphere
Based on Abdelrahman et al. formulation.
"""

import numpy as np


def sphere_gravity_forward(x: np.ndarray, z: float, x0: float, A: float) -> np.ndarray:
    """
    Calculate the vertical gravity anomaly of a 2D/3D buried sphere.

    Formula:
        g(x_i) = A * z / [ (x_i - x0)^2 + z^2 ]^(1.5)

    Parameters
    ----------
    x : np.ndarray
        Array of observation points on the profile (m).
    z : float
        Depth to the center of the sphere (m). Must be positive.
    x0 : float
        Horizontal location of the center of the sphere (m).
    A : float
        Amplitude / Scale factor (mGal * m^2).
        A = (4/3) * pi * G * delta_rho * R^3 * 1e5

    Returns
    -------
    np.ndarray
        Computed vertical gravity anomaly in mGal.
    """
    denominator = ((x - x0) ** 2 + z**2) ** 1.5
    return A * z / denominator


def calculate_amplitude_factor(radius: float, density_contrast: float) -> float:
    """
    Compute theoretical amplitude factor A from physical sphere parameters.

    Parameters
    ----------
    radius : float
        Radius of the sphere R (m).
    density_contrast : float
        Density contrast delta_rho (kg/m^3 or g/cm^3 * 1000).

    Returns
    -------
    float
        Amplitude factor A in mGal * m^2.
    """
    G = 6.67430e-11  # m^3 / (kg * s^2)
    # Factor 1e5 converts m/s^2 to mGal (1 m/s^2 = 1e5 mGal)
    A = (4.0 / 3.0) * np.pi * G * density_contrast * (radius**3) * 1e5
    return A
