"""This file contains the main class for the LMTL model."""

from __future__ import annotations

from pathlib import Path

from seapodym_lmtl_python.config import model_configuration
from seapodym_lmtl_python.config.client import close_client_locally, init_client_locally
from seapodym_lmtl_python.config.parser import parse_configuration_file
from seapodym_lmtl_python.config.saving import (
    save_configuration_locally,
    save_outputs_locally,
)
from seapodym_lmtl_python.post_production import biomass
from seapodym_lmtl_python.pre_production import pre_production
from seapodym_lmtl_python.production import production


def run_model(configuration_file_path: Path, **kwargs: dict[str, str]) -> None:
    """
    The model is run in 8 steps:
    1. Parse the configuration file.
    2. Initialize the Dask client/cluster.
    3. Generate the configuration.
    4. Run the pre-production process.
    (4.bis Save the configuration.)
    5. Run the production process.
    (5.bis Save the configuration.)
    6. Run the biomass process.
    7. Save the outputs.
    8. Close the client.

    Each step (i.e. function) is implemented in a different module.

    Parameters
    ----------
    configuration_file_path : Path
        The path to the configuration file.
        TODO(Jules): YAML only? Or any structured data?
    **kwargs : dict[str, str]
        The arguments passed to the model. They are used to override the configuration file.

    """
    # 1. PARSE (CLI integration is done here)
    parameters = parse_configuration_file(configuration_file_path, kwargs)

    # 2. INITIALIZE THE DASK CLIENT/CLUSTER
    client = init_client_locally(parameters)

    # 3. GENERATE THE CONFIGURATION
    configuration = model_configuration.process(parameters)

    # 4. RUN THE PRE-PRODUCTION PROCESS
    configuration = pre_production.process(client, configuration)

    # 4.bis SAVE THE CONFIGURATION ?
    save_configuration_locally(configuration)

    # 5. RUN THE PRODUCTION PROCESS

    configuration = production.process(configuration)

    # 5.bis SAVE THE CONFIGURATION ?
    save_configuration_locally(configuration)

    # 6. RUN BIOMASS PROCESS
    configuration = biomass.process(client, configuration)

    # 7. SAVE THE OUTPUTS
    save_outputs_locally(configuration)

    # 8. CLOSE THE CLIENT
    close_client_locally(client)
