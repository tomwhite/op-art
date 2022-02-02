import numpy as np
import op_art as opart
import op_art as xp
from numpy.testing import assert_array_equal

def test_add():
    opart.reset_ids()

    a = xp.ones((1, 2))
    b = xp.ones((1, 2))
    c = xp.add(a, b)

    assert_array_equal(c.arr, np.ones((1, 2)) + np.ones((1, 2)))
    assert_array_equal(c.src_arr_ids, [[[0, 1], [0, 1]]])
    assert_array_equal(c.src_offsets, [[[0, 0], [1, 1]]])

def test_add_broadcast():
    opart.reset_ids()

    a = xp.ones((1, 2))
    b = xp.ones((1,))
    c = xp.add(a, b)

    assert_array_equal(c.arr, np.ones((1, 2)) + np.ones((1, 2)))
    assert_array_equal(c.src_arr_ids, [[[2, 3], [2, 3]]])
    assert_array_equal(c.src_offsets, [[[0, 0], [1, 1]]])

def test_negative():
    opart.reset_ids()

    a = xp.arange(6)
    b = xp.negative(a)

    assert_array_equal(b.arr, np.negative(np.arange(6)))
    assert_array_equal(b.src_arr_ids, [[0], [0], [0], [0], [0], [0]])
    assert_array_equal(b.src_offsets, [[0], [1], [2], [3], [4], [5]])
