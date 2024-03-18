from dask.distributed import Client

from seapodym_lmtl_python.configuration.no_transport.environment import ClientParameter


class TestClientParameter:
    def test_default_client(self):
        client_param = ClientParameter()
        assert client_param.client is None
        client_param.initialize_client()
        assert isinstance(client_param.client, Client)
        client_param.close_client()
        assert client_param.client is None

    def test_from_address(self):
        client = Client(n_workers=1, threads_per_worker=1, memory_limit="1GiB")
        client_param = ClientParameter.from_address(client.scheduler.address)
        assert isinstance(client_param.client, Client)
        assert client_param.n_workers == 1
        assert client_param.threads_per_worker == 1
        assert client_param.memory_limit == 1024**3  # KiB <- 1024 B ; KB <- 1000 B
        client_param.close_client()
        assert client_param.client is None
        assert client is not None
