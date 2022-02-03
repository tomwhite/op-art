import numpy as np
from op_art import array_context
import op_art as xp
from numpy.testing import assert_array_equal

@array_context()
def test_matmul_1x1():
    a = xp.asarray([1, 2, 3])
    b = xp.asarray([4, 5, 6])
    c = xp.matmul(a, b)

    assert_array_equal(c.arr, np.matmul(np.array([1, 2, 3]), np.array([4, 5, 6])))
    assert_array_equal(c.arr, [32])
    assert_array_equal(c.src_arr_ids, [0, 0, 0, 1, 1, 1])
    assert_array_equal(c.src_offsets, [0, 1, 2, 0, 1, 2])

@array_context()
def test_matmul_1x2():
    a = xp.asarray([1, 2, 3])
    b = xp.asarray([[5, 1], [0, 3], [2, 4]])
    c = xp.matmul(a, b)

    assert_array_equal(c.arr, np.matmul(np.array([1, 2, 3]), np.array([[5, 1], [0, 3], [2, 4]])))
    assert_array_equal(c.arr, [11, 19])
    assert_array_equal(c.src_arr_ids, [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    assert_array_equal(c.src_offsets, [[0, 1, 2, 0, 2, 4], [0, 1, 2, 1, 3, 5]])

@array_context()
def test_matmul_2x1():
    a = xp.asarray([[0, 1, 2], [3, 4, 5]])
    b = xp.asarray([1, 2, 3])
    c = xp.matmul(a, b)

    assert_array_equal(c.arr, np.matmul(np.array([[0, 1, 2], [3, 4, 5]]), np.array([1, 2, 3])))
    assert_array_equal(c.arr, [8, 26])
    assert_array_equal(c.src_arr_ids, [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    assert_array_equal(c.src_offsets, [[0, 1, 2, 0, 1, 2], [3, 4, 5, 0, 1, 2]])

@array_context()
def test_matmul_2x2():
    a = xp.asarray([[0, 1, 2], [3, 4, 5]])
    b = xp.asarray([[5, 1], [0, 3], [2, 4]])
    c = xp.matmul(a, b)

    assert_array_equal(c.arr, np.matmul(np.array([[0, 1, 2], [3, 4, 5]]), np.array([[5, 1], [0, 3], [2, 4]])))
    assert_array_equal(c.src_arr_ids, [[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]], [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]])
    assert_array_equal(c.src_offsets, [[[0, 1, 2, 0, 2, 4], [0, 1, 2, 1, 3, 5]], [[3, 4, 5, 0, 2, 4], [3, 4, 5, 1, 3, 5]]])

@array_context()
def test_tensordot():
    a = xp.arange(60)
    a = xp.reshape(a, (3, 4, 5))
    b = xp.arange(24)
    b = xp.reshape(b, (4, 3, 2))
    c = xp.tensordot(a, b, axes=([1, 0], [0, 1]))

    a = np.arange(60).reshape(3,4,5)
    b = np.arange(24).reshape(4,3,2)
    assert_array_equal(c.arr, np.tensordot(a, b, axes=([1, 0], [0, 1])))

@array_context()
def test_matrix_transpose():
    a = xp.arange(6)
    a = xp.reshape(a, (3, 2))
    b = xp.matrix_transpose(a)

    assert_array_equal(b.arr, np.transpose(np.arange(6).reshape(3, 2)))
    assert_array_equal(b.src_arr_ids, [[[1], [1], [1]], [[1], [1], [1]]])
    assert_array_equal(b.src_offsets, [[[0], [2], [4]], [[1], [3], [5]]])


