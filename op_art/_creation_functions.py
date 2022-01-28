# Creation Functions
# https://data-apis.org/array-api/latest/API_specification/creation_functions.html

import numpy as np

from ._array_object import Array

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
    return Array(arr, x.arr_ids, x.offsets)

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
    return Array(arr, x.arr_ids, x.offsets)

def linspace(start, stop, /, num, *, dtype=None, device=None, endpoint=True):
    arr = np.linspace(start, stop, num, dtype=dtype, endpoint=endpoint)
    return Array(arr)

def meshgrid(*arrays, indexing="xy"):
    arr_list = np.meshgrid(*[a.arr for a in arrays], indexing=indexing)
    src_arr_ids_list = np.meshgrid(*[a.arr_ids for a in arrays], indexing=indexing)
    src_offsets_list = np.meshgrid(*[a.offsets for a in arrays], indexing=indexing)
    return [Array(a, i, o) for a, i, o in zip(arr_list, src_arr_ids_list, src_offsets_list)]

def ones(shape, *, dtype=None, device=None):
    arr = np.ones(shape, dtype)
    return Array(arr)

def ones_like(x, /, *, dtype=None, device=None):
    arr = np.ones_like(x.arr, dtype)
    return Array(arr, x.arr_ids, x.offsets)

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
    return Array(arr, src_arr_ids, src_offsets)

def tril(x, /, *, k=0):
    return _tri(x, lower=True, k=k)

def triu(x, /, *, k=0):
    return _tri(x, lower=False, k=k)

def zeros(shape, *, dtype=None, device=None):
    arr = np.zeros(shape, dtype)
    return Array(arr)

def zeros_like(x, /, *, dtype=None, device=None):
    arr = np.zeros_like(x.arr, dtype)
    return Array(arr, x.arr_ids, x.offsets)
