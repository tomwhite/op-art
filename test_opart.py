from dataclasses import asdict

import numpy as np
import op_art as opart
import op_art as xp
import pytest
from numpy.testing import assert_array_equal


def test_rewrite_representation():
    opart.reset_ids()

    a = xp.arange(6)
    b = xp.reshape(a, (3, 2))

    from op_art._visualization import rewrite_representation

    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0, 0), value=0, sources=['0_0']
    )

    # make 'a' invisible and check sources is updated
    rep = rewrite_representation(b.representation, visible_ids=[b.id])
    assert asdict(rep.cells[0]) == dict(
        id="1_0", index=(0, 0), value=0, sources=None
    )

@pytest.mark.xfail(reason="Implement in viz code")
def test_4d_not_supported():
    opart.reset_ids()

    with pytest.raises(NotImplementedError):
        xp.ones((2, 2, 2, 2))


def test_arange():
    opart.reset_ids()

    a = xp.arange(6)

    assert_array_equal(a.arr, np.arange(6))
    assert a.representation.ndim == 1
    assert a.representation.shape == (6,)
    assert len(a.representation.cells) == 6
    assert asdict(a.representation.cells[0]) == dict(
        id="0_0", index=(0,), value=0, sources=None
    )


def test_ones():
    opart.reset_ids()

    a = xp.ones((1, 2))

    assert_array_equal(a.arr, np.ones((1, 2)))
    assert len(a.representation.cells) == 2
    assert asdict(a.representation.cells[0]) == dict(
        id="0_0", index=(0, 0), value=1.0, sources=None
    )

def test_meshgrid():
    opart.reset_ids()

    a = xp.asarray([1, 2, 3, 4])
    b = xp.asarray([5, 6, 7])
    c, d = xp.meshgrid(a, b)

    assert_array_equal(c.arr, np.meshgrid(a.arr, b.arr)[0])
    assert_array_equal(d.arr, np.meshgrid(a.arr, b.arr)[1])

    assert c.representation.ndim == 2
    assert c.representation.shape == (3, 4)
    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=1, sources=["0_0"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="2_1", index=(0, 1), value=2, sources=["0_1"]
    )   
    assert asdict(c.representation.cells[4]) == dict(
        id="2_4", index=(1, 0), value=1, sources=["0_0"]
    )

    assert d.representation.ndim == 2
    assert d.representation.shape == (3, 4)
    assert asdict(d.representation.cells[0]) == dict(
        id="3_0", index=(0, 0), value=5, sources=["1_0"]
    )
    assert asdict(d.representation.cells[1]) == dict(
        id="3_1", index=(0, 1), value=5, sources=["1_0"]
    )   
    assert asdict(d.representation.cells[4]) == dict(
        id="3_4", index=(1, 0), value=6, sources=["1_1"]
    )

def test_tril():
    opart.reset_ids()

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = xp.tril(a)

    assert b.representation.ndim == 2
    assert b.representation.shape == (3, 3)
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0, 0), value=1, sources=["0_0"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_1", index=(0, 1), value=0, sources=None
    )   
    assert asdict(b.representation.cells[3]) == dict(
        id="1_3", index=(1, 0), value=4, sources=["0_3"]
    )

def test_triu():
    opart.reset_ids()

    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = xp.triu(a)

    assert b.representation.ndim == 2
    assert b.representation.shape == (3, 3)
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0, 0), value=1, sources=["0_0"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_1", index=(0, 1), value=2, sources=["0_1"]
    )   
    assert asdict(b.representation.cells[3]) == dict(
        id="1_3", index=(1, 0), value=0, sources=None
    )   

def test_tril_3d():
    opart.reset_ids()

    a = xp.arange(1, 19)
    a = xp.reshape(a, (2, 3, 3))
    b = xp.tril(a)

    assert_array_equal(b.arr, np.tril(np.arange(1, 19).reshape(2, 3, 3)))

    assert b.representation.ndim == 3
    assert b.representation.shape == (2, 3, 3)
    assert asdict(b.representation.cells[0]) == dict(
        id="2_0", index=(0, 0, 0), value=1, sources=["1_0"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="2_1", index=(0, 0, 1), value=0, sources=["1_1"]
    )   
    assert asdict(b.representation.cells[3]) == dict(
        id="2_3", index=(0, 1, 0), value=4, sources=["1_3"]
    )

def test_single_axis_indexing():
    opart.reset_ids()

    a = xp.arange(6)
    b = a[1:3]

    assert_array_equal(b.arr, np.arange(6)[1:3])
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0,), value=1, sources=["0_1"]
    )

def test_boolean_indexing():
    opart.reset_ids()

    a = xp.arange(6)
    b = xp.asarray([True, False, True, False, True, False])
    c = a[b]

    assert_array_equal(c.arr, a.arr[b.arr])
    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0,), value=0, sources=["0_0"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="2_1", index=(1,), value=2, sources=["0_2"]
    )

def test_reshape_and_index():
    opart.reset_ids()

    a = xp.arange(6)

    b = xp.reshape(a, (3, 2))

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)))
    assert b.representation.ndim == 2
    assert b.representation.shape == (3, 2)
    assert len(b.representation.cells) == 6
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0, 0), value=0, sources=["0_0"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_1", index=(0, 1), value=1, sources=["0_1"]
    )

    c = b[:, 1]

    assert_array_equal(c.arr, np.arange(6).reshape((3, 2))[:, 1])
    assert c.representation.ndim == 1
    assert c.representation.shape == (3,)
    assert len(c.representation.cells) == 3
    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0,), value=1, sources=["1_1"]
    )

def test_reshape_after_flip():
    opart.reset_ids()

    a = xp.arange(6)
    b = xp.reshape(a, (3, 2))

    c = xp.flip(b, axis=0)

    assert asdict(c.representation.cells[1]) == dict(
        id="2_1", index=(0, 1), value=5, sources=["1_5"]
    )

    d = xp.reshape(c, 6)

    assert asdict(d.representation.cells[1]) == dict(
        id="3_1", index=(1,), value=5, sources=["2_1"]
    )

def test_setitem():
    opart.reset_ids()

    a = xp.arange(3)

    b = xp.ones((3, 2))

    b[:, 1] = a

    b2 = np.ones((3, 2))
    b2[:, 1] = np.arange(3)
    assert_array_equal(b.arr, b2)

    assert b.representation.shape == (3, 2)
    assert len(b.representation.cells) == 6
    assert asdict(b.representation.cells[1]) == dict(
        id="1_1", index=(0, 1), value=0, sources=["0_0"]
    )

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

    assert c.representation.shape == (3, 2)
    assert len(c.representation.cells) == 6
    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=2, sources=["0_0", "1_0"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="2_1", index=(0, 1), value=0, sources=["3_0"]
    )

def test_concat():
    opart.reset_ids()

    a = xp.asarray([[1, 2], [3, 4]])
    b = xp.asarray([[5], [6]])
    c = xp.concat((a, b), axis=1)

    assert_array_equal(c.arr, np.concatenate((np.array([[1, 2], [3, 4]]), np.array([[5], [6]])), axis=1))

    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=1, sources=["0_0"]
    )

def test_concat_no_axis():
    opart.reset_ids()

    a = xp.asarray([[1, 2], [3, 4]])
    b = xp.asarray([[5], [6]])
    c = xp.concat((a, b), axis=None)

    assert_array_equal(c.arr, np.concatenate((np.array([[1, 2], [3, 4]]), np.array([[5], [6]])), axis=None))

    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0,), value=1, sources=["0_0"]
    )

def test_flip():
    opart.reset_ids()

    a = xp.arange(6)

    b = xp.reshape(a, (3, 2))

    c = xp.flip(b, axis=0)

    assert_array_equal(c.arr, np.flip(np.arange(6).reshape((3, 2)), 0))
    assert c.representation.ndim == 2
    assert c.representation.shape == (3, 2)
    assert len(c.representation.cells) == 6

    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=4, sources=["1_4"]
    )

def test_roll():
    opart.reset_ids()

    a = xp.arange(6)

    b = xp.roll(a, 2)

    assert_array_equal(b.arr, np.roll(np.arange(6), 2))
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0,), value=4, sources=["0_4"]
    )

def test_stack():
    opart.reset_ids()

    a = xp.asarray([1, 2, 3])
    b = xp.asarray([2, 3, 4])
    c = xp.stack((a, b))

    assert_array_equal(c.arr, np.stack((np.array([1, 2, 3]), np.array([2, 3, 4]))))

    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=1, sources=["0_0"]
    )

def test_transpose_attr():
    opart.reset_ids()

    a = xp.arange(6)

    b = xp.reshape(a, (3, 2))

    c = b.T

    assert_array_equal(c.arr, np.arange(6).reshape((3, 2)).T)
    assert c.representation.ndim == 2
    assert c.representation.shape == (2, 3)
    assert len(c.representation.cells) == 6

    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=0, sources=["1_0"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="2_1", index=(0, 1), value=2, sources=["1_2"]
    )

def test_add_ones():
    opart.reset_ids()

    a = xp.ones((1, 2))

    b = xp.ones((1, 2))

    c = xp.add(a, b)

    assert_array_equal(c.arr, np.add(np.ones((1, 2)), np.ones((1, 2))))
    assert c.representation.ndim == 2
    assert c.representation.shape == (1, 2)
    assert len(c.representation.cells) == 2

    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=2, sources=["0_0", "1_0"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="2_1", index=(0, 1), value=2, sources=["0_1", "1_1"]
    )


def test_add_ones_array():
    opart.reset_ids()

    a = xp.ones((1, 2))

    b = xp.ones((1, 2))

    c = a + b

    assert_array_equal(c.arr, np.add(np.ones((1, 2)), np.ones((1, 2))))
    assert c.representation.ndim == 2
    assert c.representation.shape == (1, 2)
    assert len(c.representation.cells) == 2

    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=2, sources=["0_0", "1_0"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="2_1", index=(0, 1), value=2, sources=["0_1", "1_1"]
    )

def test_add_ones_broadcast():
    opart.reset_ids()

    a = xp.ones((1, 2))

    b = xp.ones((1,))

    c = xp.add(a, b)

    assert_array_equal(c.arr, np.add(np.ones((1, 2)), np.ones((1,))))
    assert c.representation.ndim == 2
    assert c.representation.shape == (1, 2)
    assert len(c.representation.cells) == 2

    assert asdict(c.representation.cells[0]) == dict(
        id="4_0", index=(0, 0), value=2, sources=["2_0", "3_0"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="4_1", index=(0, 1), value=2, sources=["2_1", "3_1"]
    )

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

def test_broadcast_to():
    opart.reset_ids()

    a = xp.ones((1,))

    b = xp.broadcast_to(a, (1, 2))

    assert_array_equal(b.arr, np.broadcast_to(a.arr, (1, 2)))
    assert b.representation.ndim == 2
    assert b.representation.shape == (1, 2)
    assert len(b.representation.cells) == 2

    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0, 0), value=1, sources=["0_0"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_1", index=(0, 1), value=1, sources=["0_0"]
    )


def test_negative():
    opart.reset_ids()

    a = xp.arange(6)

    b = xp.negative(a)

    assert_array_equal(b.arr, np.negative(np.arange(6)))
    assert b.representation.ndim == 1
    assert b.representation.shape == (6,)
    assert len(b.representation.cells) == 6

    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0,), value=0, sources=["0_0"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_1", index=(1,), value=-1, sources=["0_1"]
    )


def test_sum_no_axis():
    print(np.arange(6).reshape((3, 2)).sum())
    opart.reset_ids()

    a = xp.arange(6)
    a = xp.reshape(a, (3, 2))

    b = xp.sum(a)

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)).sum())
    assert b.representation.ndim == 0
    assert b.representation.shape == ()
    assert len(b.representation.cells) == 1

    assert asdict(b.representation.cells[0]) == dict(
        id="2_0", index=(), value=15, sources=["1_0", "1_1", "1_2", "1_3", "1_4", "1_5"]
    )

def test_sum_single_axis():
    opart.reset_ids()

    a = xp.arange(6)
    a = xp.reshape(a, (3, 2))

    b = xp.sum(a, axis=0)

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)).sum(axis=0))
    assert b.representation.ndim == 1
    assert b.representation.shape == (2,)
    assert len(b.representation.cells) == 2

    assert asdict(b.representation.cells[0]) == dict(
        id="2_0", index=(0,), value=6, sources=["1_0", "1_2", "1_4"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="2_1", index=(1,), value=9, sources=["1_1", "1_3", "1_5"]
    )

def test_sum_multiple_axes():
    opart.reset_ids()

    a = xp.arange(24)
    a = xp.reshape(a, (3, 2, 4))

    b = xp.sum(a, axis=(0, 2))

    assert_array_equal(b.arr, np.arange(24).reshape((3, 2, 4)).sum(axis=(0, 2)))
    assert b.representation.ndim == 1
    assert b.representation.shape == (2,)
    assert len(b.representation.cells) == 2

def test_sum_keepdims():
    opart.reset_ids()

    a = xp.arange(6)
    a = xp.reshape(a, (3, 2))

    b = xp.sum(a, axis=0, keepdims=True)

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)).sum(axis=0, keepdims=True))
    assert b.representation.ndim == 2
    assert b.representation.shape == (1, 2)
    assert len(b.representation.cells) == 2

    assert asdict(b.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=6, sources=["1_0", "1_2", "1_4"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="2_1", index=(0, 1), value=9, sources=["1_1", "1_3", "1_5"]
    )


def test_mean():
    opart.reset_ids()

    a = xp.arange(6, dtype=xp.float32)
    a = xp.reshape(a, (3, 2))

    b = xp.mean(a, axis=0)

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)).mean(axis=0))
    assert b.representation.ndim == 1
    assert b.representation.shape == (2,)
    assert len(b.representation.cells) == 2

    assert asdict(b.representation.cells[0]) == dict(
        id="2_0", index=(0,), value=2, sources=["1_0", "1_2", "1_4"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="2_1", index=(1,), value=3, sources=["1_1", "1_3", "1_5"]
    )


def test_argsort():
    opart.reset_ids()

    a = xp.asarray([5, 1, 0, 3, 2, 4])
    b = xp.argsort(a)

    print(b)

    assert len(b.representation.cells) == 6
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0,), value=2, sources=["0_2"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_1", index=(1,), value=1, sources=["0_1"]
    )


def test_sort():
    opart.reset_ids()

    a = xp.asarray([5, 1, 0, 3, 2, 4])
    b = xp.sort(a)

    assert len(b.representation.cells) == 6
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0,), value=0, sources=["0_2"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_1", index=(1,), value=1, sources=["0_1"]
    )


def test_unique_values():
    opart.reset_ids()

    a = xp.asarray([2, 2, 3, 5, 1, 1])
    b = xp.unique_values(a)

    assert len(b.representation.cells) == 4
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0,), value=1, sources=["0_4", "0_5"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_1", index=(1,), value=2, sources=["0_0", "0_1"]
    )
    assert asdict(b.representation.cells[2]) == dict(
        id="1_2", index=(2,), value=3, sources=["0_2"]
    )

def test_unique_counts():
    opart.reset_ids()

    a = xp.asarray([2, 2, 3, 5, 1, 1])
    b, c = xp.unique_counts(a)

    assert len(b.representation.cells) == 4
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0,), value=1, sources=["0_4", "0_5"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_1", index=(1,), value=2, sources=["0_0", "0_1"]
    )
    assert asdict(b.representation.cells[2]) == dict(
        id="1_2", index=(2,), value=3, sources=["0_2"]
    )

    assert len(c.representation.cells) == 4
    assert asdict(c.representation.cells[0]) == dict(
        id="4_0", index=(0,), value=2, sources=["0_4", "0_5"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="4_1", index=(1,), value=2, sources=["0_0", "0_1"]
    )
    assert asdict(c.representation.cells[2]) == dict(
        id="4_2", index=(2,), value=1, sources=["0_2"]
    )

def test_unique_inverse():
    opart.reset_ids()

    a = xp.asarray([2, 2, 3, 5, 1, 1])
    b, c = xp.unique_inverse(a)

    assert len(b.representation.cells) == 4
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0,), value=1, sources=["0_4", "0_5"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_1", index=(1,), value=2, sources=["0_0", "0_1"]
    )
    assert asdict(b.representation.cells[2]) == dict(
        id="1_2", index=(2,), value=3, sources=["0_2"]
    )

    assert len(c.representation.cells) == 6
    assert asdict(c.representation.cells[0]) == dict(
        id="3_0", index=(0,), value=1, sources=["0_0"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="3_1", index=(1,), value=1, sources=["0_1"]
    )
    assert asdict(c.representation.cells[2]) == dict(
        id="3_2", index=(2,), value=2, sources=["0_2"]
    )

def test_unique_all():
    opart.reset_ids()

    a = xp.asarray([2, 2, 3, 5, 1, 1])
    b, c, d, e = xp.unique_all(a)

    assert len(b.representation.cells) == 4
    assert len(c.representation.cells) == 4
    assert len(d.representation.cells) == 6
    assert len(e.representation.cells) == 4

    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0,), value=4, sources=["0_4", "0_5"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="2_1", index=(1,), value=0, sources=["0_0", "0_1"]
    )
    assert asdict(c.representation.cells[2]) == dict(
        id="2_2", index=(2,), value=2, sources=["0_2"]
    )

def test_unique_2d():
    opart.reset_ids()

    a = xp.asarray([2, 2, 3, 5, 1, 1])
    a = xp.reshape(a, (2, 3))
    b, c, d, e = xp.unique_all(a)

    assert len(b.representation.cells) == 4
    assert len(c.representation.cells) == 4
    assert len(d.representation.cells) == 6
    assert len(e.representation.cells) == 4

    assert b.shape == (4,)
    assert c.shape == (4,)
    assert d.shape == (2, 3)
    assert e.shape == (4,)

    assert asdict(c.representation.cells[0]) == dict(
        id="3_0", index=(0,), value=4, sources=["1_4", "1_5"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="3_1", index=(1,), value=0, sources=["1_0", "1_1"]
    )
    assert asdict(c.representation.cells[2]) == dict(
        id="3_2", index=(2,), value=2, sources=["1_2"]
    )

    assert len(b.representation.cells) == 4
    assert asdict(b.representation.cells[0]) == dict(
        id="2_0", index=(0,), value=1, sources=["1_4", "1_5"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="2_1", index=(1,), value=2, sources=["1_0", "1_1"]
    )
    assert asdict(b.representation.cells[2]) == dict(
        id="2_2", index=(2,), value=3, sources=["1_2"]
    )

def test_einsum():
    opart.reset_ids()

    a = xp.asarray([[1, 2], [3, 4]])
    b = xp.asarray([[1, 2], [3, 4]])
    c = xp.einsum("ij,jk->ik", a, b)

    assert c.shape == (2, 2)
    assert len(c.representation.cells) == 4
    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=7,
        sources=["0_0", "0_1", "1_0", "1_2"]
    )

def test_matmul():
    opart.reset_ids()

    a = xp.asarray([[0, 1, 2], [3, 4, 5]])
    b = xp.asarray([[5, 1], [0, 3], [2, 4]])
    c = xp.matmul(a, b)

    assert c.shape == (2, 2)
    assert len(c.representation.cells) == 4
    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=4,
        sources=["0_0", "0_1", "0_2", "1_0", "1_2", "1_4"]
    )

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

    assert b.shape == (2, 3)
    assert asdict(b.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=0,
        sources=["1_0"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="2_1", index=(0, 1), value=2,
        sources=["1_2"]
    )

    assert_array_equal(b.arr, np.transpose(np.arange(6).reshape(3, 2)))

def test_argmax():
    opart.reset_ids()

    a = xp.arange(6)
    a = xp.reshape(a, (3, 2))

    b = xp.argmax(a, axis=0)

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)).argmax(axis=0))
    assert b.representation.ndim == 1
    assert b.representation.shape == (2,)
    assert len(b.representation.cells) == 2

    assert asdict(b.representation.cells[0]) == dict(
        id="2_0", index=(0,), value=2, sources=["1_0", "1_2", "1_4"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="2_1", index=(1,), value=2, sources=["1_1", "1_3", "1_5"]
    )

def test_nonzero():
    opart.reset_ids()

    a = xp.asarray([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    b, c = xp.nonzero(a)

    assert b.representation.ndim == 1
    assert b.representation.shape == (4,)
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0,), value=0, sources=["0_0"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_1", index=(1,), value=1, sources=["0_4"]
    )

    assert c.representation.ndim == 1
    assert c.representation.shape == (4,)
    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0,), value=0, sources=["0_0"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="2_1", index=(1,), value=1, sources=["0_4"]
    )

def test_where():
    opart.reset_ids()

    a = xp.asarray([True, False, True, True])
    b = xp.asarray([1, 2, 3, 4])
    c = xp.asarray([9, 8, 7, 6])
    d = xp.where(a, b, c)

    assert d.representation.ndim == 1
    assert d.representation.shape == (4,)
    assert asdict(d.representation.cells[0]) == dict(
        id="3_0", index=(0,), value=1, sources=["0_0", "1_0"]
    )
    assert asdict(d.representation.cells[1]) == dict(
        id="3_1", index=(1,), value=8, sources=["0_1", "2_1"]
    )   
