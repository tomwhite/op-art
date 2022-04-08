import numpy as np
from numpy.testing import assert_array_equal

import op_art as xp
from op_art import array_context


@array_context()
def test_unique_values():
    a = xp.asarray([2, 2, 3, 5, 1, 1])
    b = xp.unique_values(a)

    assert_array_equal(b.arr, np.unique(np.array([2, 2, 3, 5, 1, 1])))
    assert_array_equal(b.arr, [1, 2, 3, 5])
    assert_array_equal(b.src_arr_ids, [[0, 0], [0, 0], [0, -1], [0, -1]])
    assert_array_equal(b.src_offsets, [[4, 5], [0, 1], [2, -1], [3, -1]])


@array_context()
def test_unique_counts():
    a = xp.asarray([2, 2, 3, 5, 1, 1])
    b, c = xp.unique_counts(a)

    u0, u1 = np.unique(np.array([2, 2, 3, 5, 1, 1]), return_counts=True)
    assert_array_equal(b.arr, u0)
    assert_array_equal(c.arr, u1)

    assert_array_equal(b.arr, [1, 2, 3, 5])
    assert_array_equal(b.src_arr_ids, [[0, 0], [0, 0], [0, -1], [0, -1]])
    assert_array_equal(b.src_offsets, [[4, 5], [0, 1], [2, -1], [3, -1]])

    assert_array_equal(c.arr, [2, 2, 1, 1])
    assert_array_equal(c.src_arr_ids, [[0, 0], [0, 0], [0, -1], [0, -1]])
    assert_array_equal(c.src_offsets, [[4, 5], [0, 1], [2, -1], [3, -1]])


@array_context()
def test_unique_inverse():
    a = xp.asarray([2, 2, 3, 5, 1, 1])
    b, c = xp.unique_inverse(a)

    u0, u1 = np.unique(np.array([2, 2, 3, 5, 1, 1]), return_inverse=True)
    assert_array_equal(b.arr, u0)
    assert_array_equal(c.arr, u1)

    assert_array_equal(b.arr, [1, 2, 3, 5])
    assert_array_equal(b.src_arr_ids, [[0, 0], [0, 0], [0, -1], [0, -1]])
    assert_array_equal(b.src_offsets, [[4, 5], [0, 1], [2, -1], [3, -1]])

    assert_array_equal(c.arr, [1, 1, 2, 3, 0, 0])
    assert_array_equal(c.src_arr_ids, [[0], [0], [0], [0], [0], [0]])
    assert_array_equal(c.src_offsets, [[0], [1], [2], [3], [4], [5]])


@array_context()
def test_unique_all():
    a = xp.asarray([2, 2, 3, 5, 1, 1])
    b, c, d, e = xp.unique_all(a)

    u0, u1, u2, u3 = np.unique(
        np.array([2, 2, 3, 5, 1, 1]),
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    assert_array_equal(b.arr, u0)
    assert_array_equal(c.arr, u1)
    assert_array_equal(d.arr, u2)
    assert_array_equal(e.arr, u3)

    assert_array_equal(b.arr, [1, 2, 3, 5])
    assert_array_equal(b.src_arr_ids, [[0, 0], [0, 0], [0, -1], [0, -1]])
    assert_array_equal(b.src_offsets, [[4, 5], [0, 1], [2, -1], [3, -1]])

    assert_array_equal(c.arr, [4, 0, 2, 3])
    assert_array_equal(c.src_arr_ids, [[0, 0], [0, 0], [0, -1], [0, -1]])
    assert_array_equal(c.src_offsets, [[4, 5], [0, 1], [2, -1], [3, -1]])

    assert_array_equal(d.arr, [1, 1, 2, 3, 0, 0])
    assert_array_equal(d.src_arr_ids, [[0], [0], [0], [0], [0], [0]])
    assert_array_equal(d.src_offsets, [[0], [1], [2], [3], [4], [5]])

    assert_array_equal(e.arr, [2, 2, 1, 1])
    assert_array_equal(e.src_arr_ids, [[0, 0], [0, 0], [0, -1], [0, -1]])
    assert_array_equal(e.src_offsets, [[4, 5], [0, 1], [2, -1], [3, -1]])


@array_context()
def test_unique_2d():
    a = xp.asarray([2, 2, 3, 5, 1, 1])
    a = xp.reshape(a, (2, 3))
    b, c, d, e = xp.unique_all(a)

    u0, u1, u2, u3 = np.unique(
        np.array([2, 2, 3, 5, 1, 1]),
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    u2 = u2.reshape(a.shape)

    assert_array_equal(b.arr, [1, 2, 3, 5])
    assert_array_equal(b.src_arr_ids, [[1, 1], [1, 1], [1, -1], [1, -1]])
    assert_array_equal(b.src_offsets, [[4, 5], [0, 1], [2, -1], [3, -1]])

    assert_array_equal(c.arr, [4, 0, 2, 3])
    assert_array_equal(c.src_arr_ids, [[1, 1], [1, 1], [1, -1], [1, -1]])
    assert_array_equal(c.src_offsets, [[4, 5], [0, 1], [2, -1], [3, -1]])

    assert_array_equal(d.arr, [[1, 1, 2], [3, 0, 0]])
    assert_array_equal(d.src_arr_ids, [[[1], [1], [1]], [[1], [1], [1]]])
    assert_array_equal(d.src_offsets, [[[0], [1], [2]], [[3], [4], [5]]])

    assert_array_equal(e.arr, [2, 2, 1, 1])
    assert_array_equal(c.src_arr_ids, [[1, 1], [1, 1], [1, -1], [1, -1]])
    assert_array_equal(e.src_offsets, [[4, 5], [0, 1], [2, -1], [3, -1]])
