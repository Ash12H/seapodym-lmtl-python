from time import time

from seapopym.function.generator.cell_area import cell_area_kernel
from seapopym.logging.custom_logger import logger


class TestCellArea:
    def test_cell_area_no_chunk(self, state_preprod_fg4_t4d_y1_x1_z3):
        start = time()
        kernel = cell_area_kernel()
        res = kernel.run(state_preprod_fg4_t4d_y1_x1_z3)
        stop = time()
        logger.debug(f"Execution time no chunk: {stop - start}")
        assert res.chunks is None

    def test_cell_area_chunk(self, state_preprod_fg4_t4d_y1_x1_z3):
        chunk = {"Y": 1, "X": 1}
        start = time()
        kernel = cell_area_kernel(chunk=chunk)
        res_chunked = kernel.run(state_preprod_fg4_t4d_y1_x1_z3.cf.chunk(chunk))
        _ = res_chunked.compute()
        stop = time()
        logger.debug(f"Execution time all chunk: {stop - start}")
        assert res_chunked.chunks == ((1,), (1,))  # dim y = 1, dim x = 1
