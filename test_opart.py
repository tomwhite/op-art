from dataclasses import asdict

import numpy as np
import op_art as opart
import pytest
from numpy.testing import assert_array_equal


def test_rewrite_representation():
    opart.reset_ids()

    a = opart.arange(6)
    b = opart.reshape(a, (3, 2))

    from op_art._array_object import rewrite_representation

    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0, 0), value=0, sources=['0_0']
    )

    # make 'a' invisible and check sources is updated
    rep = rewrite_representation(b.representation, visible_ids=[b.id])
    assert asdict(rep.cells[0]) == dict(
        id="1_0", index=(0, 0), value=0, sources=None
    )

def test_4d_not_supported():
    opart.reset_ids()

    with pytest.raises(NotImplementedError):
        opart.ones((2, 2, 2, 2))


def test_arange():
    opart.reset_ids()

    a = opart.arange(6)

    assert_array_equal(a.arr, np.arange(6))
    assert a.representation.ndim == 1
    assert a.representation.shape == (6,)
    assert len(a.representation.cells) == 6
    assert asdict(a.representation.cells[0]) == dict(
        id="0_0", index=(0,), value=0, sources=None
    )


def test_ones():
    opart.reset_ids()

    a = opart.ones((1, 2))

    assert_array_equal(a.arr, np.ones((1, 2)))
    assert len(a.representation.cells) == 2
    assert asdict(a.representation.cells[0]) == dict(
        id="0_0", index=(0, 0), value=1.0, sources=None
    )

def test_single_axis_indexing():
    opart.reset_ids()

    a = opart.arange(6)
    b = a[1:3]

    assert_array_equal(b.arr, np.arange(6)[1:3])
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0,), value=1, sources=["0_8"]
    )

def test_reshape_and_index():
    opart.reset_ids()

    a = opart.arange(6)

    b = opart.reshape(a, (3, 2))

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)))
    assert b.representation.ndim == 2
    assert b.representation.shape == (3, 2)
    assert len(b.representation.cells) == 6
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0, 0), value=0, sources=["0_0"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_8", index=(0, 1), value=1, sources=["0_8"]
    )

    c = b[:, 1]

    assert_array_equal(c.arr, np.arange(6).reshape((3, 2))[:, 1])
    assert c.representation.ndim == 1
    assert c.representation.shape == (3,)
    assert len(c.representation.cells) == 3
    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0,), value=1, sources=["1_8"]
    )

def test_reshape_after_flip():
    opart.reset_ids()

    a = opart.arange(6)
    b = opart.reshape(a, (3, 2))
    c = opart.flip(b, axis=0)

    assert asdict(c.representation.cells[1]) == dict(
        id="2_-24", index=(2, 1), value=1, sources=["1_8"]
    )

    d = opart.reshape(c, 6)

    assert asdict(d.representation.cells[5]) == dict(
        id="3_40", index=(5,), value=1, sources=["2_-24"]
    )

def test_setitem():
    opart.reset_ids()

    a = opart.arange(3)

    b = opart.ones((3, 2))

    b[:, 1] = a

    b2 = np.ones((3, 2))
    b2[:, 1] = np.arange(3)
    assert_array_equal(b.arr, b2)

    assert b.representation.shape == (3, 2)
    assert len(b.representation.cells) == 6
    assert asdict(b.representation.cells[1]) == dict(
        id="1_8", index=(0, 1), value=0, sources=["0_0"]
    )

def test_concat():
    opart.reset_ids()

    a = opart.asarray([[1, 2], [3, 4]])
    b = opart.asarray([[5], [6]])
    c = opart.concat((a, b), axis=1)

    assert_array_equal(c.arr, np.concatenate((np.array([[1, 2], [3, 4]]), np.array([[5], [6]])), axis=1))

    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=1, sources=["0_0"]
    )

def test_flip():
    opart.reset_ids()

    a = opart.arange(6)

    b = opart.reshape(a, (3, 2))

    c = opart.flip(b, axis=0)

    assert_array_equal(c.arr, np.flip(np.arange(6).reshape((3, 2)), 0))
    assert c.representation.ndim == 2
    assert c.representation.shape == (3, 2)
    assert len(c.representation.cells) == 6

    assert asdict([cell for cell in c.representation.cells if cell.index == (0, 0)][0]) == dict(
        id="2_0", index=(0, 0), value=4, sources=["1_32"]
    )

def test_roll():
    opart.reset_ids()

    a = opart.arange(6)

    b = opart.roll(a, 2)

    assert_array_equal(b.arr, np.roll(np.arange(6), 2))
    assert asdict([cell for cell in b.representation.cells if cell.index == (0,)][0]) == dict(
        id="5_0", index=(0,), value=4, sources=["2_0"]
    )

def test_stack():
    opart.reset_ids()

    a = opart.asarray([1, 2, 3])
    b = opart.asarray([2, 3, 4])
    c = opart.stack((a, b))

    assert_array_equal(c.arr, np.stack((np.array([1, 2, 3]), np.array([2, 3, 4]))))

    assert asdict(c.representation.cells[0]) == dict(
        id="4_0", index=(0, 0), value=1, sources=["2_0"]
    )

def test_transpose_attr():
    opart.reset_ids()

    a = opart.arange(6)

    b = opart.reshape(a, (3, 2))

    c = b.T

    assert_array_equal(c.arr, np.arange(6).reshape((3, 2)).T)
    assert c.representation.ndim == 2
    assert c.representation.shape == (2, 3)
    assert len(c.representation.cells) == 6

    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=0, sources=["1_0"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="2_8", index=(1, 0), value=1, sources=["1_8"]
    )

def test_add_ones():
    opart.reset_ids()

    a = opart.ones((1, 2))

    b = opart.ones((1, 2))

    c = opart.add(a, b)

    assert_array_equal(c.arr, np.add(np.ones((1, 2)), np.ones((1, 2))))
    assert c.representation.ndim == 2
    assert c.representation.shape == (1, 2)
    assert len(c.representation.cells) == 2

    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=2, sources=["0_0", "1_0"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="2_8", index=(0, 1), value=2, sources=["0_8", "1_8"]
    )


def test_add_ones_broadcast():
    opart.reset_ids()

    a = opart.ones((1, 2))

    b = opart.ones((1,))

    c = opart.add(a, b)

    assert_array_equal(c.arr, np.add(np.ones((1, 2)), np.ones((1,))))
    assert c.representation.ndim == 2
    assert c.representation.shape == (1, 2)
    assert len(c.representation.cells) == 2

    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=2, sources=["0_0", "1_0"]
    )
    assert asdict(c.representation.cells[1]) == dict(
        id="2_8", index=(0, 1), value=2, sources=["0_8", "1_0"]
    )


def test_broadcast_to():
    opart.reset_ids()

    a = opart.ones((1,))

    b = opart.broadcast_to(a, (1, 2))

    assert_array_equal(b.arr, np.broadcast_to(a.arr, (1, 2)))
    assert b.representation.ndim == 2
    assert b.representation.shape == (1, 2)
    assert len(b.representation.cells) == 2

    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0, 0), value=1, sources=["0_0"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_8", index=(0, 1), value=1, sources=["0_0"]
    )


def test_negative():
    opart.reset_ids()

    a = opart.arange(6)

    b = opart.negative(a)

    assert_array_equal(b.arr, np.negative(np.arange(6)))
    assert b.representation.ndim == 1
    assert b.representation.shape == (6,)
    assert len(b.representation.cells) == 6

    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0,), value=0, sources=["0_0"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_8", index=(1,), value=-1, sources=["0_8"]
    )


def test_sum():
    opart.reset_ids()

    a = opart.arange(6)
    a = opart.reshape(a, (3, 2))

    b = opart.sum(a, axis=0)

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)).sum(axis=0))
    assert b.representation.ndim == 1
    assert b.representation.shape == (2,)
    assert len(b.representation.cells) == 2

    assert asdict(b.representation.cells[0]) == dict(
        id="2_0", index=(0,), value=6, sources=["1_0", "1_16", "1_32"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="2_8", index=(1,), value=9, sources=["1_8", "1_24", "1_40"]
    )


def test_mean():
    opart.reset_ids()

    a = opart.arange(6)
    a = opart.reshape(a, (3, 2))

    b = opart.mean(a, axis=0)

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)).mean(axis=0))
    assert b.representation.ndim == 1
    assert b.representation.shape == (2,)
    assert len(b.representation.cells) == 2

    assert asdict(b.representation.cells[0]) == dict(
        id="2_0", index=(0,), value=2, sources=["1_0", "1_16", "1_32"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="2_8", index=(1,), value=3, sources=["1_8", "1_24", "1_40"]
    )


def test_argsort():
    opart.reset_ids()

    a = opart.asarray([5, 1, 0, 3, 2, 4])
    b = opart.argsort(a)

    assert len(b.representation.cells) == 6
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0,), value=2, sources=["0_16"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_8", index=(1,), value=1, sources=["0_8"]
    )


def test_sort():
    opart.reset_ids()

    a = opart.asarray([5, 1, 0, 3, 2, 4])
    b = opart.sort(a)

    assert len(b.representation.cells) == 6
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0,), value=0, sources=["0_16"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_8", index=(1,), value=1, sources=["0_8"]
    )


def test_unique():
    opart.reset_ids()

    a = opart.asarray([2, 2, 3, 5, 1, 1])
    b = opart.unique(a)

    assert len(b.representation.cells) == 4
    assert asdict(b.representation.cells[0]) == dict(
        id="1_0", index=(0,), value=1, sources=["0_32"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="1_8", index=(1,), value=2, sources=["0_0"]
    )

def test_einsum():
    opart.reset_ids()

    a = opart.asarray([[1, 2], [3, 4]])
    b = opart.asarray([[1, 2], [3, 4]])
    c = opart.einsum("ij,jk->ik", a, b)

    assert c.shape == (2, 2)
    assert len(c.representation.cells) == 4
    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=7,
        sources=["0_0", "0_8", "1_0", "1_16"]
    )

def test_matmul():
    opart.reset_ids()

    a = opart.asarray([[0, 1, 2], [3, 4, 5]])
    b = opart.asarray([[5, 1], [0, 3], [2, 4]])
    c = opart.matmul(a, b)

    assert c.shape == (2, 2)
    assert len(c.representation.cells) == 4
    assert asdict(c.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=4,
        sources=["0_0", "0_8", "0_16", "1_0", "1_16", "1_32"]
    )

def test_tensordot():
    opart.reset_ids()

    a = opart.arange(60)
    a = opart.reshape(a, (3, 4, 5))
    b = opart.arange(24)
    b = opart.reshape(b, (4, 3, 2))
    c = opart.tensordot(a, b, axes=([1, 0], [0, 1]))

    a = np.arange(60).reshape(3,4,5)
    b = np.arange(24).reshape(4,3,2)
    assert_array_equal(c.arr, np.tensordot(a, b, axes=([1, 0], [0, 1])))

def test_transpose():
    opart.reset_ids()

    a = opart.arange(6)
    a = opart.reshape(a, (3, 2))
    b = opart.transpose(a)

    assert b.shape == (2, 3)
    assert asdict(b.representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=0,
        sources=["1_0"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="2_8", index=(1, 0), value=1,
        sources=["1_8"]
    )

    assert_array_equal(b.arr, np.transpose(np.arange(6).reshape(3, 2)))

def test_argmax():
    opart.reset_ids()

    a = opart.arange(6)
    a = opart.reshape(a, (3, 2))

    b = opart.argmax(a, axis=0)

    assert_array_equal(b.arr, np.arange(6).reshape((3, 2)).argmax(axis=0))
    assert b.representation.ndim == 1
    assert b.representation.shape == (2,)
    assert len(b.representation.cells) == 2

    assert asdict(b.representation.cells[0]) == dict(
        id="2_0", index=(0,), value=2, sources=["1_0", "1_16", "1_32"]
    )
    assert asdict(b.representation.cells[1]) == dict(
        id="2_8", index=(1,), value=2, sources=["1_8", "1_24", "1_40"]
    )