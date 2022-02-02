# Searching Functions
# https://data-apis.org/array-api/latest/API_specification/searching_functions.html

import numpy as np

from ._array_object import Array, _normalize_two_args, _reduction_operation
from ._data_type_functions import broadcast_arrays

def argmax(x, /, *, axis=None, keepdims=False):
    xp = x.arr.__array_namespace__()
    return _reduction_operation(x, axis, xp.argmax, keepdims=keepdims)

def argmin(x, /, *, axis=None, keepdims=False):
    xp = x.arr.__array_namespace__()
    return _reduction_operation(x, axis, xp.argmin, keepdims=keepdims)

def nonzero(x, /):
    xp = x.arr.__array_namespace__()
    arrs = xp.nonzero(x.arr)
    if x.ndim == 0:
        src_arr_ids = None
        src_offsets = None
    else:
        # we would like to say:
        # src_arr_ids = x.arr_ids[arrs]
        # src_offsets = x.offsets[arrs]
        # but the array api does not allow integer array indices, so use np
        np_arrs = tuple(np.asarray(a) for a in arrs)
        src_arr_ids = xp.asarray(np.asarray(x.arr_ids)[np_arrs])
        src_offsets = xp.asarray(np.asarray(x.offsets)[np_arrs])
    return tuple(Array(arr, src_arr_ids, src_offsets) for arr in arrs)

def where(condition, x1, x2, /):
    xp = condition.arr.__array_namespace__()
    #x1, x2 = _normalize_two_args(x1, x2)
    arr = xp.where(condition.arr, x1.arr, x2.arr)

    # broadcast if necessary
    condition_broad, x1_broad, x2_broad = broadcast_arrays(condition, x1, x2)

    # use np.where on the source arrays
    x1_or_x2_arr_ids = xp.where(condition.arr, x1_broad.arr_ids, x2_broad.arr_ids)
    src_arr_ids = xp.stack([condition_broad.arr_ids, x1_or_x2_arr_ids], axis=-1)

    x1_or_x2_offsets = xp.where(condition.arr, x1_broad.offsets, x2_broad.offsets)
    src_offsets = xp.stack([condition_broad.offsets, x1_or_x2_offsets], axis=-1)

    return Array(arr, src_arr_ids, src_offsets)