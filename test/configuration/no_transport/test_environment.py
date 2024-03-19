from pathlib import Path

from dask.distributed import Client

from seapodym_lmtl_python.configuration.no_transport.parameter_environment import (
    BaseOuputForcingParameter,
    BiomassParameter,
    ChunkParameter,
    ClientParameter,
    EnvironmentParameter,
    OutputParameter,
    PreProductionParameter,
    ProductionParameter,
)


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


class TestChunkParameter:
    def test_as_dict_with_fgroup(self):
        expected_chunks = {"functional_group": "auto"}
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


class TestBaseOuputForcingParameter:
    def test_default_values(self):
        param = BaseOuputForcingParameter()
        assert param.path == Path("./output.nc")
        assert param.with_parameter is True
        assert param.with_forcing is False

    def test_custom_values(self):
        path = "/path/to/outputs"
        with_parameter = False
        with_forcing = True

        param = BaseOuputForcingParameter(path=path, with_parameter=with_parameter, with_forcing=with_forcing)
        assert param.path == Path(path)
        assert param.with_parameter is with_parameter
        assert param.with_forcing is with_forcing


class TestPreProductionParameter:
    def test_default_values(self):
        param = PreProductionParameter()
        assert param.path == Path("./output.nc")
        assert param.with_parameter is True
        assert param.with_forcing is False
        assert param.timestamps == [-1]

    def test_custom_values(self):
        path = "/path/to/outputs"
        with_parameter = False
        with_forcing = True
        timestamps = [0, 1, 2]

        param = PreProductionParameter(
            path=path, with_parameter=with_parameter, with_forcing=with_forcing, timestamps=timestamps
        )
        assert param.path == Path(path)
        assert param.with_parameter is with_parameter
        assert param.with_forcing is with_forcing
        assert param.timestamps == timestamps


class TestOutputParameter:
    def test_default_values(self):
        output_param = OutputParameter()
        assert output_param.biomass is not None
        assert isinstance(output_param.biomass, BiomassParameter)
        assert output_param.production is not None
        assert isinstance(output_param.production, ProductionParameter)
        assert output_param.pre_production is not None
        assert isinstance(output_param.pre_production, PreProductionParameter)

    def test_shared_path_as_default(self):
        output_param = OutputParameter(
            biomass=BiomassParameter(),
            production=ProductionParameter(),
            pre_production=PreProductionParameter(),
        )
        assert output_param.shared_path()

    def test_not_shared_path(self):
        output_param = OutputParameter(
            biomass=BiomassParameter(path="/path/to/outputs1"),
            production=ProductionParameter(path="/path/to/outputs2"),
            pre_production=PreProductionParameter(path="/path/to/outputs3"),
        )

        assert not output_param.shared_path()


class TestEnvironmentParameter:
    def test_default_values(self):
        param = EnvironmentParameter()
        assert param.chunk is not None
        assert isinstance(param.chunk, ChunkParameter)
        assert param.client is not None
        assert isinstance(param.client, ClientParameter)
        assert param.output is not None
        assert isinstance(param.output, OutputParameter)

    def test_custom_values(self):
        chunk_param = ChunkParameter()
        client_param = ClientParameter()
        output_param = OutputParameter()

        param = EnvironmentParameter(chunk=chunk_param, client=client_param, output=output_param)
        assert param.chunk is chunk_param
        assert param.client is client_param
        assert param.output is output_param
