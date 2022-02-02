import numpy as np
import op_art as opart
import op_art as xp
from numpy.testing import assert_array_equal

def test_concat():
    opart.reset_ids()

    a = xp.asarray([[1, 2], [3, 4]])
    b = xp.asarray([[5], [6]])
    c = xp.concat((a, b), axis=1)

    assert_array_equal(c.arr, np.concatenate((np.array([[1, 2], [3, 4]]), np.array([[5], [6]])), axis=1))
    assert_array_equal(c.src_arr_ids, [[0, 0, 1], [0, 0, 1]])
    assert_array_equal(c.src_offsets, [[0, 1, 0], [2, 3, 1]])

def test_concat_no_axis():
    opart.reset_ids()

    a = xp.asarray([[1, 2], [3, 4]])
    b = xp.asarray([[5], [6]])
    c = xp.concat((a, b), axis=None)

    assert_array_equal(c.arr, np.concatenate((np.array([[1, 2], [3, 4]]), np.array([[5], [6]])), axis=None))
    assert_array_equal(c.src_arr_ids, [0, 0, 0, 0, 1, 1])
    assert_array_equal(c.src_offsets, [0, 1, 2, 3, 0, 1])

def test_flip():
    opart.reset_ids()

    a = xp.arange(6)
    b = xp.reshape(a, (3, 2))
    c = xp.flip(b, axis=0)

    assert_array_equal(c.arr, np.flip(np.arange(6).reshape((3, 2)), axis=0))
    assert_array_equal(c.src_arr_ids, [[1, 1], [1, 1], [1, 1]])
    assert_array_equal(c.src_offsets, [[4, 5], [2, 3], [0, 1]])

def test_flip_reshape():
    opart.reset_ids()

    a = xp.arange(6)
    b = xp.reshape(a, (3, 2))
    c = xp.flip(b, axis=0)
    d = xp.reshape(c, 6)

    assert_array_equal(d.arr, np.flip(np.arange(6).reshape((3, 2)), axis=0).reshape(6))
    assert_array_equal(d.src_arr_ids, [2, 2, 2, 2, 2, 2])
    assert_array_equal(d.src_offsets, [0, 1, 2, 3, 4, 5])

def test_roll():
    opart.reset_ids()

    a = xp.arange(6)
    b = xp.roll(a, 2)

    assert_array_equal(b.arr, np.roll(np.arange(6), 2))
    assert_array_equal(b.src_offsets, [4, 5, 0, 1, 2, 3])

def test_stack():
    opart.reset_ids()

    a = xp.asarray([1, 2, 3])
    b = xp.asarray([2, 3, 4])
    c = xp.stack((a, b))

    assert_array_equal(c.arr, np.stack((np.array([1, 2, 3]), np.array([2, 3, 4]))))

    assert_array_equal(c.src_arr_ids, [[0, 0, 0], [1, 1, 1]])
    assert_array_equal(c.src_offsets, [[0, 1, 2], [0, 1, 2]])
