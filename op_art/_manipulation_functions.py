# Manipulation Functions
# https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html

import numpy as np

from ._array_object import Array, _structural_operation
from ._data_type_functions import result_type

def concat(arrays, /, *, axis=0):
    dtype = result_type(*arrays)
    arr = np.concatenate([a.arr for a in arrays], axis=axis, dtype=dtype)
    src_arr_ids = np.concatenate([a.arr_ids for a in arrays], axis=axis)
    src_offsets = np.concatenate([a.offsets for a in arrays], axis=axis)
    return Array(arr, src_arr_ids, src_offsets)

def expand_dims(x, /, *, axis):
    return _structural_operation(x, np.expand_dims, axis=axis)

def flip(x, /, *, axis=None):
    return _structural_operation(x, np.flip, axis=axis)

def permute_dims(x, /, axes):
    return _structural_operation(x, np.transpose, axes=axes)

def reshape(x, /, shape):
    return _structural_operation(x, np.reshape, shape)

def roll(x, /, shift, *, axis=None):
    return _structural_operation(x, np.roll, shift, axis=axis)

def squeeze(x, /, axis):
    return _structural_operation(x, np.squeeze, axis=axis)

def stack(arrays, /, *, axis=0):
    arr = np.stack([a.arr for a in arrays], axis=axis)
    src_arr_ids = np.stack([a.arr_ids for a in arrays], axis=axis)
    src_offsets = np.stack([a.offsets for a in arrays], axis=axis)
    return Array(arr, src_arr_ids, src_offsets)
