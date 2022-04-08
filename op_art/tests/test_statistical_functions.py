import numpy as np
from op_art import array_context
import op_art as xp
from numpy.testing import assert_array_equal


@array_context()
def test_sum_no_axis():
    a = xp.arange(6)
    a = xp.reshape(a, (3, 2))

    b = xp.sum(a)

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)).sum())
    assert_array_equal(b.src_arr_ids, [1, 1, 1, 1, 1, 1])
    assert_array_equal(b.src_offsets, [0, 1, 2, 3, 4, 5])


@array_context()
def test_sum_single_axis():
    a = xp.arange(6)
    a = xp.reshape(a, (3, 2))

    b = xp.sum(a, axis=0)

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)).sum(axis=0))
    assert_array_equal(b.src_arr_ids, [[1, 1, 1], [1, 1, 1]])
    assert_array_equal(b.src_offsets, [[0, 2, 4], [1, 3, 5]])


@array_context()
def test_sum_multiple_axes():
    a = xp.arange(24)
    a = xp.reshape(a, (3, 2, 4))

    b = xp.sum(a, axis=(0, 2))

    assert_array_equal(b.arr, np.arange(24).reshape((3, 2, 4)).sum(axis=(0, 2)))
    assert_array_equal(b.arr, [114, 162])
    assert_array_equal(
        b.src_arr_ids,
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
    )
    assert_array_equal(
        b.src_offsets,
        [
            [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19],
            [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23],
        ],
    )


@array_context()
def test_sum_keepdims():
    a = xp.arange(6)
    a = xp.reshape(a, (3, 2))

    b = xp.sum(a, axis=0, keepdims=True)

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)).sum(axis=0, keepdims=True))
    assert_array_equal(b.src_arr_ids, [[[1, 1, 1], [1, 1, 1]]])
    assert_array_equal(b.src_offsets, [[[0, 2, 4], [1, 3, 5]]])


@array_context()
def test_mean():
    a = xp.arange(6, dtype=xp.float32)
    a = xp.reshape(a, (3, 2))

    b = xp.mean(a, axis=0)

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)).mean(axis=0))
    assert_array_equal(b.src_arr_ids, [[1, 1, 1], [1, 1, 1]])
    assert_array_equal(b.src_offsets, [[0, 2, 4], [1, 3, 5]])
