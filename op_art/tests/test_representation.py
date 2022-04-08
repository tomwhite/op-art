from dataclasses import asdict

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import op_art as xp
from op_art import array_context
from op_art._visualization import _get_representation, rewrite_representation


@array_context()
def test_representation():
    a = xp.arange(6)

    assert_array_equal(a.arr, np.arange(6))

    representation = _get_representation(a)

    assert representation.ndim == 1
    assert representation.shape == (6,)
    assert len(representation.cells) == 6
    assert asdict(representation.cells[0]) == dict(
        id="0_0", index=(0,), value=0, sources=None
    )


@array_context()
def test_representation_multiple_sources():
    a = xp.ones((1, 2))
    b = xp.ones((1, 2))
    c = xp.add(a, b)

    assert_array_equal(c.arr, np.add(np.ones((1, 2)), np.ones((1, 2))))

    representation = _get_representation(c)

    assert representation.ndim == 2
    assert representation.shape == (1, 2)
    assert len(representation.cells) == 2

    assert asdict(representation.cells[0]) == dict(
        id="2_0", index=(0, 0), value=2, sources=["0_0", "1_0"]
    )
    assert asdict(representation.cells[1]) == dict(
        id="2_1", index=(0, 1), value=2, sources=["0_1", "1_1"]
    )


@array_context()
def test_rewrite_representation():
    a = xp.arange(6)
    b = xp.reshape(a, (3, 2))

    representation = _get_representation(b)

    assert asdict(representation.cells[0]) == dict(
        id="1_0", index=(0, 0), value=0, sources=["0_0"]
    )

    # make 'a' invisible and check sources is updated
    rep = rewrite_representation(representation, visible_ids=[b.id])
    assert asdict(rep.cells[0]) == dict(id="1_0", index=(0, 0), value=0, sources=None)


@pytest.mark.xfail(reason="Implement in viz code")
@array_context()
def test_4d_not_supported():
    with pytest.raises(NotImplementedError):
        xp.ones((2, 2, 2, 2))
