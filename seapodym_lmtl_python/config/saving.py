"""
This module implements the function to save the configuration using Zarr. It can be done either locally or on a
remote storage like S3 or google cloud.
"""

import xarray as xr


def save_configuration_locally(configuration: xr.Dataset) -> None:
    """
    Save the configuration locally using Zarr.

    Parameters
    ----------
    configuration : xr.Dataset
        The model configuration that contains both forcing and parameters.

    """
    # NOTE(Jules): Optional: Zarr + done in parallel?


def save_outputs_locally(outputs: xr.Dataset) -> None:
    """
    Save the outputs locally using Zarr.

    Parameters
    ----------
    outputs : xr.Dataset
        The model outputs.

    """
    # NOTE(Jules): Optional : Add forcing (T, NPP) and computed forcings (Daylenght, landmask, cell_area) to the output.
