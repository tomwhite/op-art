import numpy as np
from numpy.testing import assert_array_equal

import op_art as xp
from op_art import array_context


@array_context()
def test_arange():
    a = xp.arange(6)

    assert_array_equal(a.arr, np.arange(6))
    assert np.all(a.arr_ids == 0)
    assert_array_equal(a.offsets, [0, 1, 2, 3, 4, 5])
    assert a.src_arr_ids is None
    assert a.src_offsets is None


@array_context()
def test_meshgrid():
    a = xp.asarray([1, 2, 3, 4])
    b = xp.asarray([5, 6, 7])
    c, d = xp.meshgrid(a, b)

    assert_array_equal(c.arr, np.meshgrid(a.arr, b.arr)[0])
    assert_array_equal(d.arr, np.meshgrid(a.arr, b.arr)[1])

    assert_array_equal(
        np.asarray(c.src_arr_ids).squeeze(), [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    assert_array_equal(
        np.asarray(c.src_offsets).squeeze(), [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    )

    assert_array_equal(
        np.asarray(d.src_arr_ids).squeeze(), [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    )
    assert_array_equal(
        np.asarray(d.src_offsets).squeeze(), [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]
    )


@array_context()
def test_ones():
    a = xp.ones((1, 2))

    assert_array_equal(a.arr, np.ones((1, 2)))
    assert np.all(a.arr_ids == 0)
    assert_array_equal(a.offsets, [[0, 1]])
    assert a.src_arr_ids is None
    assert a.src_offsets is None


@array_context()
def test_tril():
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = xp.tril(a)

    assert_array_equal(b.arr, np.tril(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
    assert_array_equal(
        np.asarray(b.src_arr_ids).squeeze(), [[0, -1, -1], [0, 0, -1], [0, 0, 0]]
    )
    assert_array_equal(
        np.asarray(b.src_offsets).squeeze(), [[0, -1, -1], [3, 4, -1], [6, 7, 8]]
    )


@array_context()
def test_triu():
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = xp.triu(a)

    assert_array_equal(b.arr, np.triu(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
    assert_array_equal(
        np.asarray(b.src_arr_ids).squeeze(), [[0, 0, 0], [-1, 0, 0], [-1, -1, 0]]
    )
    assert_array_equal(
        np.asarray(b.src_offsets).squeeze(), [[0, 1, 2], [-1, 4, 5], [-1, -1, 8]]
    )


@array_context()
def test_tril_3d():
    a = xp.arange(1, 19)
    a = xp.reshape(a, (2, 3, 3))
    b = xp.tril(a)

    assert_array_equal(b.arr, np.tril(np.arange(1, 19).reshape(2, 3, 3)))
    assert_array_equal(
        np.asarray(b.src_arr_ids).squeeze(),
        [[[1, -1, -1], [1, 1, -1], [1, 1, 1]], [[1, -1, -1], [1, 1, -1], [1, 1, 1]]],
    )
    assert_array_equal(
        np.asarray(b.src_offsets).squeeze(),
        [
            [[0, -1, -1], [3, 4, -1], [6, 7, 8]],
            [[9, -1, -1], [12, 13, -1], [15, 16, 17]],
        ],
    )
