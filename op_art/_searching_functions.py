# Searching Functions
# https://data-apis.org/array-api/latest/API_specification/searching_functions.html

import numpy as np

from ._array_object import Array, _normalize_two_args, _reduction_operation
from ._data_type_functions import broadcast_arrays

def argmax(x, /, *, axis=None, keepdims=False):
    return _reduction_operation(x, axis, np.argmax, keepdims=keepdims)

def argmin(x, /, *, axis=None, keepdims=False):
    return _reduction_operation(x, axis, np.argmin, keepdims=keepdims)

def nonzero(x, /):
    arrs = np.nonzero(x.arr)
    if x.ndim == 0:
        src_arr_ids = None
        src_offsets = None
    else:
        src_arr_ids = x.arr_ids[arrs]
        src_offsets = x.offsets[arrs]
    return tuple(Array(arr, src_arr_ids, src_offsets) for arr in arrs)

def where(condition, x1, x2, /):
    x1, x2 = _normalize_two_args(x1, x2)
    arr = np.where(condition.arr, x1.arr, x2.arr)

    # broadcast if necessary
    condition_broad, x1_broad, x2_broad = broadcast_arrays(condition, x1, x2)

    # use np.where on the source arrays
    x1_or_x2_arr_ids = np.where(condition.arr, x1_broad.arr_ids, x2_broad.arr_ids)
    src_arr_ids = np.stack([condition_broad.arr_ids, x1_or_x2_arr_ids], axis=-1)

    x1_or_x2_offsets = np.where(condition.arr, x1_broad.offsets, x2_broad.offsets)
    src_offsets = np.stack([condition_broad.offsets, x1_or_x2_offsets], axis=-1)

    return Array(arr, src_arr_ids, src_offsets)