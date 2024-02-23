"""All the tools needed to manage the Dask client."""

from dask.distributed import Client

from seapodym_lmtl_python.config import Parameters


def init_client_locally(param: Parameters) -> Client:
    """
    Initialize the dask client locally.

    Parameters
    ----------
    param : Parameters
        The parameters of the model.

    Returns
    -------
    Client
        The dask client.

    """
    # NOTE(Jules): So many arguments can be passed to the Client class. This can be setup in the configuration file or
    # with CLI. CLI should override (i.e. max priority) the configuration file.


def close_client_locally(client: Client) -> None:
    """
    Close the dask client locally.

    Parameters
    ----------
    client : Client
        The dask client.

    """
