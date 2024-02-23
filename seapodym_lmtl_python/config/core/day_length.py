"""This module contains functions used to calculate the length of the day."""

import numpy as np
import pint


def day_length_forsythe(latitude: float, day_of_the_year: int, p: int = 0) -> float:
    """
    Compute the day length for a given latitude, day of the year and twilight angle.

    NOTE Jules : Seapodym fish (CSimtunaFunc::daylength_twilight)
    The CBM model of Forsythe et al, Ecological Modelling 80 (1995) 87-95
    p - angle between the sun position and the horizon, in degrees :
        - 6  => civil twilight
        - 12 => nautical twilight
        - 18 => astronomical twilight
    """
    # revolution angle for the day of the year
    theta = 0.2163108 + 2 * np.arctan(
        0.9671396 * np.tan(0.00860 * (day_of_the_year - 186))
    )
    # sun's declination angle, or the angular distance at solar noon between the
    # Sun and the equator, from the Eartch orbit revolution angle
    phi = np.arcsin(0.39795 * np.cos(theta))
    # daylength computed according to 'p'
    arg = (np.sin(np.pi * p / 180) + np.sin(latitude * np.pi / 180) * np.sin(phi)) / (
        np.cos(latitude * np.pi / 180) * np.cos(phi)
    )

    arg = np.clip(arg, -1.0, 1.0)

    return (24.0 - (24.0 / np.pi) * np.arccos(arg)) * pint.application_registry.hour
