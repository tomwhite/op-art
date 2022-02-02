import numpy as np
import op_art as opart
import op_art as xp
from numpy.testing import assert_array_equal

def test_broadcast_to():
    opart.reset_ids()

    a = xp.ones((1,))
    b = xp.broadcast_to(a, (1, 2))

    assert np.all(b.arr_ids == 1)
    assert_array_equal(b.offsets, [[0, 1]])
    assert_array_equal(b.src_arr_ids, [[0, 0]])
    assert_array_equal(b.src_offsets, [[0, 0]])
