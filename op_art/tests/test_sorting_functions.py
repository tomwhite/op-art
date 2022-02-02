import numpy as np
import op_art as opart
import op_art as xp
from numpy.testing import assert_array_equal

def test_argsort():
    opart.reset_ids()

    a = xp.asarray([5, 1, 0, 3, 2, 4])
    b = xp.argsort(a)

    assert_array_equal(b.arr, np.argsort(np.array([5, 1, 0, 3, 2, 4])))
    assert_array_equal(b.src_arr_ids, [[0], [0], [0], [0], [0], [0]])
    assert_array_equal(b.src_offsets, [[2], [1], [4], [3], [5], [0]])

def test_sort():
    opart.reset_ids()

    a = xp.asarray([5, 1, 0, 3, 2, 4])
    b = xp.sort(a)

    assert_array_equal(b.arr, np.sort(np.array([5, 1, 0, 3, 2, 4])))
    assert_array_equal(b.src_arr_ids, [[0], [0], [0], [0], [0], [0]])
    assert_array_equal(b.src_offsets, [[2], [1], [4], [3], [5], [0]])
