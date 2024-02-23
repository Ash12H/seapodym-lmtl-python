"""
This module contains the class and functions used to manage the configuration of the model.

If you plan to use the model in a **standart way**, you should only use the `model_configuration` function.

If you want to use the model in a more **advanced way**, you can use the `FunctionParameters`, `PathParameters`,
`FunctionalGroupUnit`, `FunctionalGroups` and `Parameters` classes to manage the forcings and the parameters of the
model. In this case, you can inherit from these classes to benefit from the validation and the documentation.
"""

from .parameters import (
    FunctionParameters,
    PathParameters,
    FunctionalGroupUnit,
    FunctionalGroups,
    Parameters,
)

from .model_configuration import model_configuration

from .parser import parse_configuration_file

from .saving import save_configuration_locally, save_outputs_locally

from .client import init_client_locally, close_client_locally
