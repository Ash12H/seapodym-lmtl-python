import numpy as np

from seapopym.function.generator.production.compiled_functions import expand_dims


class TestExpandDims:
    def test_1d_array(self):
        data = np.array([1, 2, 3])
        expanded_data = expand_dims(data, 2)
        assert expanded_data.shape == (3, 2)
        assert np.array_equal(expanded_data[..., 0], data)
        assert np.array_equal(expanded_data[..., 1], np.zeros(3))

    def test_2d_array(self):
        data = np.array([[1, 2], [3, 4]])
        expanded_data = expand_dims(data, 3)
        assert expanded_data.shape == (2, 2, 3)
        assert np.array_equal(expanded_data[..., 0], data)
        assert np.array_equal(expanded_data[..., 1], np.zeros((2, 2)))
        assert np.array_equal(expanded_data[..., 2], np.zeros((2, 2)))
