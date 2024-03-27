from time import time

from seapopym.function.generator.cell_area import cell_area
from seapopym.logging.custom_logger import logger


class TestCellArea:
    def test_cell_area_no_chunk(self, state_preprod_fg4_t4d_y1_x1_z3):
        start = time()
        res = cell_area(state_preprod_fg4_t4d_y1_x1_z3)
        stop = time()
        logger.debug(f"Execution time no chunk: {stop - start}")
        assert res.chunks is None

    def test_cell_area_chunk(self, state_preprod_fg4_t4d_y1_x1_z3):
        chunk = {"Y": 1, "X": 1}
        start = time()
        res_chunked = cell_area(state_preprod_fg4_t4d_y1_x1_z3.cf.chunk(chunk), chunk=chunk)
        _ = res_chunked.compute()
        stop = time()
        logger.debug(f"Execution time all chunk: {stop - start}")
        assert res_chunked.chunks == ((1,), (1,))  # dim y = 1, dim x = 1

    def test_cell_area_chunk_only_template(self, state_preprod_fg4_t4d_y1_x1_z3):
        chunk = {"Y": 1}
        start = time()
        res = cell_area(state_preprod_fg4_t4d_y1_x1_z3, chunk=chunk)
        stop = time()
        logger.debug(f"Execution time only template chunk: {stop - start}")
        assert res.chunks is None
