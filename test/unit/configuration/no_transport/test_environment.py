from pathlib import Path

from dask.distributed import Client

from seapopym.configuration.parameters.parameter_environment import (
    ChunkParameter,
    ClientParameter,
    EnvironmentParameter
)


class TestClientParameter:
    def test_default_client(self):
        client_param = ClientParameter()
        assert client_param.client is None
        client_param.initialize_client()
        assert isinstance(client_param.client, Client)
        client_param.close_client()
        assert client_param.client is None
        client_param.initialize_client()
        assert isinstance(client_param.client, Client)
        client_param.close_client()

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


class TestChunkParameter:
    def test_as_dict_with_fgroup(self):
        expected_chunks = {"functional_group": 1}
        chunk_param = ChunkParameter()
        assert chunk_param.as_dict() == expected_chunks

        expected_chunks = {"functional_group": 10}
        chunk_param = ChunkParameter(**expected_chunks)
        assert chunk_param.as_dict() == expected_chunks

        expected_chunks = {"functional_group": 10, "latitude": "auto"}
        chunk_param = ChunkParameter(**expected_chunks)
        assert chunk_param.as_dict() == expected_chunks

        expected_chunks = {"functional_group": 10, "latitude": 10, "longitude": 10}
        chunk_param = ChunkParameter(**expected_chunks)
        assert chunk_param.as_dict() == expected_chunks

    def test_as_dict_without_fgroup(self):
        expected_chunks = {}
        chunk_param = ChunkParameter()
        assert chunk_param.as_dict(with_fgroup=False) == expected_chunks

        expected_chunks = {"latitude": "auto"}
        chunk_param = ChunkParameter(**expected_chunks)
        assert chunk_param.as_dict(with_fgroup=False) == expected_chunks

        expected_chunks = {"latitude": 10}
        chunk_param = ChunkParameter(**expected_chunks)
        assert chunk_param.as_dict(with_fgroup=False) == expected_chunks

        expected_chunks = {"latitude": 10, "longitude": 10}
        chunk_param = ChunkParameter(**expected_chunks)
        assert chunk_param.as_dict(with_fgroup=False) == expected_chunks



class TestEnvironmentParameter:
    def test_default_values(self):
        param = EnvironmentParameter()
        assert param.chunk is not None
        assert isinstance(param.chunk, ChunkParameter)
        assert param.client is not None
        assert isinstance(param.client, ClientParameter)

    def test_custom_values(self):
        chunk_param = ChunkParameter()
        client_param = ClientParameter()

        param = EnvironmentParameter(chunk=chunk_param, client=client_param)
        assert param.chunk is chunk_param
        assert param.client is client_param
