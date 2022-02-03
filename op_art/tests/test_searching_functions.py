import numpy as np
from op_art import array_context
import op_art as xp
from numpy.testing import assert_array_equal

@array_context()
def test_argmax():
    a = xp.arange(6)
    a = xp.reshape(a, (3, 2))

    b = xp.argmax(a, axis=0)

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)).argmax(axis=0))
    assert_array_equal(b.arr, [2, 2])
    assert_array_equal(b.src_arr_ids, [[1, 1, 1], [1, 1, 1]])
    assert_array_equal(b.src_offsets, [[0, 2, 4], [1, 3, 5]])

@array_context()
def test_nonzero():
    a = xp.asarray([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    b, c = xp.nonzero(a)

    nz0, nz1 = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]]).nonzero()
    assert_array_equal(b.arr, nz0)
    assert_array_equal(c.arr, nz1)

    assert_array_equal(b.arr, [0, 1, 2, 2])
    assert_array_equal(b.src_arr_ids, [[0], [0], [0], [0]])
    assert_array_equal(b.src_offsets, [[0], [4], [6], [7]])

    assert_array_equal(c.arr, [0, 1, 0, 1])
    assert_array_equal(c.src_arr_ids, [[0], [0], [0], [0]])
    assert_array_equal(c.src_offsets, [[0], [4], [6], [7]])

@array_context()
def test_where():
    a = xp.asarray([True, False, True, True])
    b = xp.asarray([1, 2, 3, 4])
    c = xp.asarray([9, 8, 7, 6])
    d = xp.where(a, b, c)

    assert_array_equal(d.arr, np.where(np.array([True, False, True, True]), np.array([1, 2, 3, 4]), np.array([9, 8, 7, 6])))

    assert_array_equal(d.arr, [1, 8, 3, 4])
    assert_array_equal(d.src_arr_ids, [[0, 1], [0, 2], [0, 1], [0, 1]])
    assert_array_equal(d.src_offsets, [[0, 0], [1, 1], [2, 2], [3, 3]])
