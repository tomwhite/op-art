# Creation Functions
# https://data-apis.org/array-api/latest/API_specification/creation_functions.html

# Note: many can delegate directly to the equivalent underlying array function unless
# they take existing arrays as input

import numpy as np

from ._array_object import Array, _direct_mapping

# device is not supported

def arange(start, /, stop=None, step=1, *, dtype=None, device=None):
    arr = np.arange(start, stop, step, dtype)
    return Array(arr)

def asarray(obj, /, *, dtype=None, device=None, copy=None):
    if isinstance(obj, Array):
        if dtype is not None and obj.dtype != dtype:
            copy = True
        if copy in (True, np._CopyMode.ALWAYS):
            return Array(np.array(obj.arr, copy=True, dtype=dtype))
        return obj
    arr = np.asarray(obj, dtype)
    return Array(arr)

def empty(shape, *, dtype=None, device=None):
    arr = np.empty(shape, dtype)
    return Array(arr)

def empty_like(x, /, *, dtype=None, device=None):
    arr = np.empty_like(x.arr, dtype)
    # TODO: the _like functions should animate x's shape, not its cells
    return _direct_mapping(x, arr)

def eye(n_rows, n_cols=None, /, *, k=0, dtype=None, device=None):
    arr = np.eye(n_rows, n_cols, k=k, dtype=dtype)
    return Array(arr)

def from_dlpack(x, /):
    arr = np._from_dlpack(x)
    return Array(arr)

def full(shape, fill_value, *, dtype=None, device=None):
    arr = np.full(shape, fill_value, dtype)
    return Array(arr)

def full_like(x, /, fill_value, *, dtype=None, device=None):
    arr = np.full_like(x.arr, fill_value, dtype)
    return _direct_mapping(x, arr)

def linspace(start, stop, /, num, *, dtype=None, device=None, endpoint=True):
    arr = np.linspace(start, stop, num, dtype=dtype, endpoint=endpoint)
    return Array(arr)

def meshgrid(*arrays, indexing="xy"):
    if len({a.dtype for a in arrays}) > 1:
        raise ValueError("meshgrid inputs must all have the same dtype")

    # based on https://github.com/numpy/numpy/blob/v1.22.0/numpy/lib/function_base.py#L4805-L4951
    import op_art as xp
    ndim = len(arrays)
    s0 = (1,) * ndim
    output = [xp.reshape(a, s0[:i] + (-1,) + s0[i + 1:])
              for i, a in enumerate(arrays)]

    if indexing == "xy" and ndim > 1:
        # switch first and second axis
        output[0] = xp.reshape(output[0], (1, -1) + s0[2:])
        output[1] = xp.reshape(output[1], (-1, 1) + s0[2:])

    return xp.broadcast_arrays(*output)

def ones(shape, *, dtype=None, device=None):
    arr = np.ones(shape, dtype)
    return Array(arr)

def ones_like(x, /, *, dtype=None, device=None):
    arr = np.ones_like(x.arr, dtype)
    return _direct_mapping(x, arr)

def _tri(x, /, *, lower, k=0):
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for tril/triu")

    if lower:
        arr = np.tril(x.arr, k=k)
        tri_indices = np.tril_indices(x.shape[-2], k=k, m=x.shape[-1])
    else:
        arr = np.triu(x.arr, k=k)
        tri_indices = np.triu_indices(x.shape[-2], k=k, m=x.shape[-1])

    src_arr_ids = np.full_like(arr, -1, dtype=np.int32)
    src_offsets = np.full_like(arr, -1, dtype=np.int32)
    if x.ndim > 2:
        src_arr_ids[:, tri_indices] = x.arr_ids[:, tri_indices]
        src_offsets[:, tri_indices] = x.offsets[:, tri_indices]
    else:
        src_arr_ids[tri_indices] = x.arr_ids[tri_indices]
        src_offsets[tri_indices] = x.offsets[tri_indices]
    return Array(arr, src_arr_ids=src_arr_ids, src_offsets=src_offsets)

def tril(x, /, *, k=0):
    return _tri(x, lower=True, k=k)

def triu(x, /, *, k=0):
    return _tri(x, lower=False, k=k)

def zeros(shape, *, dtype=None, device=None):
    arr = np.zeros(shape, dtype)
    return Array(arr)

def zeros_like(x, /, *, dtype=None, device=None):
    arr = np.zeros_like(x.arr, dtype)
    return _direct_mapping(x, arr)