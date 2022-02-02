import numpy as np
import op_art as opart
import op_art as xp
from numpy.testing import assert_array_equal

def test_transpose_attr():
    opart.reset_ids()

    a = xp.arange(6)
    b = xp.reshape(a, (3, 2))
    c = b.T

    assert_array_equal(c.arr, np.arange(6).reshape((3, 2)).T)
    assert np.all(c.arr_ids == 2)
    assert_array_equal(c.offsets, [[0, 1, 2], [3, 4, 5]])
    assert_array_equal(c.src_arr_ids, [[[1], [1], [1]], [[1], [1], [1]]])
    assert_array_equal(c.src_offsets, [[[0], [2], [4]], [[1], [3], [5]]])

def test_getitem_single_axis():
    opart.reset_ids()

    a = xp.arange(6)
    b = a[1:3]

    assert_array_equal(b.arr, np.arange(6)[1:3])
    assert np.all(b.arr_ids == 1)
    assert_array_equal(b.offsets, [0, 1])
    assert_array_equal(b.src_arr_ids, [[0], [0]])
    assert_array_equal(b.src_offsets, [[1], [2]])

def test_getitem_boolean_array():
    opart.reset_ids()

    a = xp.arange(6)
    b = xp.asarray([True, False, True, False, True, False])
    c = a[b]

    assert_array_equal(c.arr, a.arr[b.arr])
    assert np.all(c.arr_ids == 2)
    assert_array_equal(c.offsets, [0, 1, 2])
    assert_array_equal(c.src_arr_ids, [[0], [0], [0]])
    assert_array_equal(c.src_offsets, [[0], [2], [4]])

def test_setitem():
    opart.reset_ids()

    a = xp.arange(3)
    b = xp.ones((3, 2))
    b[:, 1] = a

    b2 = np.ones((3, 2))
    b2[:, 1] = np.arange(3)
    assert_array_equal(b.arr, b2)

    assert np.all(b.arr_ids == 1)
    assert_array_equal(b.offsets, [[0, 1], [2, 3], [4, 5]])
    assert_array_equal(b.src_arr_ids, [[-1, 0], [-1, 0], [-1, 0]])
    assert_array_equal(b.src_offsets, [[-1, 0], [-1, 1], [-1, 2]])

def test_setitem_multiple_sources():
    opart.reset_ids()

    a = xp.ones((3, 2))
    b = xp.ones((3, 2))
    c = xp.add(a, b)
    d = xp.arange(3)
    c[:, 1] = d

    c2 = np.ones((3, 2)) + np.ones((3, 2))
    c2[:, 1] = np.arange(3)
    assert_array_equal(c.arr, c2)

    assert np.all(c.arr_ids == 2)
    assert_array_equal(c.offsets, [[0, 1], [2, 3], [4, 5]])
    assert_array_equal(c.src_arr_ids, [[[0, 1], [3, -1]], [[0, 1], [3, -1]], [[0, 1], [3, -1]]])
    assert_array_equal(c.src_offsets, [[[0, 0], [0, -1]], [[2, 2], [1, -1]], [[4, 4], [2, -1]]])

def test_setitem_edge_case():
    opart.reset_ids()

    a = xp.asarray(False)
    b = xp.asarray(False)
    b[a] = False

    assert_array_equal(b.src_arr_ids, [-1])

def test_add():
    opart.reset_ids()

    a = xp.ones((1, 2))
    b = xp.ones((1, 2))
    c = a + b

    assert_array_equal(c.arr, np.ones((1, 2)) + np.ones((1, 2)))
    assert_array_equal(c.src_arr_ids, [[[0, 1], [0, 1]]])
    assert_array_equal(c.src_offsets, [[[0, 0], [1, 1]]])

def test_add_inplace():
    opart.reset_ids()

    a = xp.ones((1, 2))
    b = xp.ones((1, 2))

    assert np.all(a.arr_ids == 0)
    assert_array_equal(a.offsets, [[0, 1]])
    assert a.src_arr_ids == None
    assert a.src_offsets == None

    a += b

    a2 = np.ones((1, 2))
    a2 += np.ones((1, 2))
    assert_array_equal(a.arr, a2)

    # a's src arrays have been updated
    assert np.all(a.arr_ids == 0)
    assert_array_equal(a.offsets, [[0, 1]])
    assert_array_equal(a.src_arr_ids, [[[0, 1], [0, 1]]])
    assert_array_equal(a.src_offsets, [[[0, 0], [1, 1]]])