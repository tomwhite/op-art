import numpy as np
import op_art as opart
import op_art as xp
from numpy.testing import assert_array_equal

def test_matmul():
    opart.reset_ids()

    a = xp.asarray([[0, 1, 2], [3, 4, 5]])
    b = xp.asarray([[5, 1], [0, 3], [2, 4]])
    c = xp.matmul(a, b)

    assert_array_equal(c.arr, np.matmul(np.array([[0, 1, 2], [3, 4, 5]]), np.array([[5, 1], [0, 3], [2, 4]])))
    assert_array_equal(c.src_arr_ids, [[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]], [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]])
    assert_array_equal(c.src_offsets, [[[0, 1, 2, 0, 2, 4], [0, 1, 2, 1, 3, 5]], [[3, 4, 5, 0, 2, 4], [3, 4, 5, 1, 3, 5]]])

def test_tensordot():
    opart.reset_ids()

    a = xp.arange(60)
    a = xp.reshape(a, (3, 4, 5))
    b = xp.arange(24)
    b = xp.reshape(b, (4, 3, 2))
    c = xp.tensordot(a, b, axes=([1, 0], [0, 1]))

    a = np.arange(60).reshape(3,4,5)
    b = np.arange(24).reshape(4,3,2)
    assert_array_equal(c.arr, np.tensordot(a, b, axes=([1, 0], [0, 1])))

def test_matrix_transpose():
    opart.reset_ids()

    a = xp.arange(6)
    a = xp.reshape(a, (3, 2))
    b = xp.matrix_transpose(a)

    assert_array_equal(b.arr, np.transpose(np.arange(6).reshape(3, 2)))
    assert_array_equal(b.src_arr_ids, [[1, 1, 1], [1, 1, 1]])
    assert_array_equal(b.src_offsets, [[0, 2, 4], [1, 3, 5]])


