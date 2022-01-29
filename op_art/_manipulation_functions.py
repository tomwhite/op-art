# Manipulation Functions
# https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html

from ._array_object import Array, _structural_operation
from ._data_type_functions import result_type

def concat(arrays, /, *, axis=0):
    result_type(*arrays) # ensure at least one array
    xp = arrays[0].arr.__array_namespace__()
    # TODO: generalise _structural_operation to work on lists of arrays
    arr = xp.concat([a.arr for a in arrays], axis=axis)
    src_arr_ids = xp.concat([a.arr_ids for a in arrays], axis=axis)
    src_offsets = xp.concat([a.offsets for a in arrays], axis=axis)
    return Array(arr, src_arr_ids, src_offsets)

def expand_dims(x, /, *, axis):
    xp = x.arr.__array_namespace__()
    return _structural_operation(x, xp.expand_dims, axis=axis)

def flip(x, /, *, axis=None):
    xp = x.arr.__array_namespace__()
    return _structural_operation(x, xp.flip, axis=axis)

def permute_dims(x, /, axes):
    xp = x.arr.__array_namespace__()
    return _structural_operation(x, xp.permute_dims, axes=axes)

def reshape(x, /, shape):
    xp = x.arr.__array_namespace__()
    return _structural_operation(x, xp.reshape, shape)

def roll(x, /, shift, *, axis=None):
    xp = x.arr.__array_namespace__()
    return _structural_operation(x, xp.roll, shift, axis=axis)

def squeeze(x, /, axis):
    xp = x.arr.__array_namespace__()
    return _structural_operation(x, xp.squeeze, axis=axis)

def stack(arrays, /, *, axis=0):
    result_type(*arrays) # ensure at least one array
    xp = arrays[0].arr.__array_namespace__()
    arr = xp.stack([a.arr for a in arrays], axis=axis)
    src_arr_ids = xp.stack([a.arr_ids for a in arrays], axis=axis)
    src_offsets = xp.stack([a.offsets for a in arrays], axis=axis)
    return Array(arr, src_arr_ids, src_offsets)
