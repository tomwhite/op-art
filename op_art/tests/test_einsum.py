import numpy as np
from op_art import array_context
import op_art as xp
from numpy.testing import assert_array_equal

@array_context()
def test_einsum():
    a = xp.asarray([[1, 2], [3, 4]])
    b = xp.asarray([[1, 2], [3, 4]])
    c = xp.einsum("ij,jk->ik", a, b)

    assert_array_equal(c.arr, np.einsum("ij,jk->ik", np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])))
    assert_array_equal(c.src_arr_ids, [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1]]])
    assert_array_equal(c.src_offsets, [[[0, 1, 0, 2], [0, 1, 1, 3]], [[2, 3, 0, 2], [2, 3, 1, 3]]])
