# Creation Functions
# https://data-apis.org/array-api/latest/API_specification/creation_functions.html

import numpy as np
import numpy.array_api as nxp

from ._array_object import Array


def arange(start, /, stop=None, step=1, *, dtype=None, device=None):
    arr = nxp.arange(start, stop, step, dtype=dtype, device=device)
    return Array(arr)


def asarray(obj, /, *, dtype=None, device=None, copy=None):
    if isinstance(obj, Array):
        if dtype is not None and obj.dtype != dtype:
            copy = True
        if copy in (True, np._CopyMode.ALWAYS):
            return Array(nxp.asarray(obj.arr, copy=True, dtype=dtype))
        return obj
    arr = nxp.asarray(obj, dtype=dtype, device=device, copy=copy)
    return Array(arr)


def empty(shape, *, dtype=None, device=None):
    arr = nxp.empty(shape, dtype=dtype, device=device)
    return Array(arr)


def empty_like(x, /, *, dtype=None, device=None):
    arr = nxp.empty_like(x.arr, dtype=dtype, device=device)
    # TODO: the _like functions should animate x's shape, not its cells
    return Array(arr, x.arr_ids, x.offsets)


def eye(n_rows, n_cols=None, /, *, k=0, dtype=None, device=None):
    arr = nxp.eye(n_rows, n_cols, k=k, dtype=dtype, device=device)
    return Array(arr)


def from_dlpack(x, /):
    arr = nxp.from_dlpack(x)
    return Array(arr)


def full(shape, fill_value, *, dtype=None, device=None):
    arr = nxp.full(shape, fill_value, dtype=dtype, device=device)
    return Array(arr)


def full_like(x, /, fill_value, *, dtype=None, device=None):
    arr = nxp.full_like(x.arr, fill_value, dtype=dtype, device=device)
    return Array(arr, x.arr_ids, x.offsets)


def linspace(start, stop, /, num, *, dtype=None, device=None, endpoint=True):
    arr = nxp.linspace(start, stop, num, dtype=dtype, device=device, endpoint=endpoint)
    return Array(arr)


def meshgrid(*arrays, indexing="xy"):
    arr_list = nxp.meshgrid(*[a.arr for a in arrays], indexing=indexing)
    src_arr_ids_list = nxp.meshgrid(*[a.arr_ids for a in arrays], indexing=indexing)
    src_offsets_list = nxp.meshgrid(*[a.offsets for a in arrays], indexing=indexing)
    return [
        Array(a, i, o) for a, i, o in zip(arr_list, src_arr_ids_list, src_offsets_list)
    ]


def ones(shape, *, dtype=None, device=None):
    arr = nxp.ones(shape, dtype=dtype, device=device)
    return Array(arr)


def ones_like(x, /, *, dtype=None, device=None):
    arr = nxp.ones_like(x.arr, dtype=dtype, device=device)
    return Array(arr, x.arr_ids, x.offsets)


def _tri(x, /, *, lower, k=0):
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for tril/triu")

    xp = x.arr.__array_namespace__()
    mask = xp.ones((x.shape[-2], x.shape[-1]), dtype=xp.int32)
    if lower:
        arr = xp.tril(x.arr, k=k)
        mask = xp.tril(mask, k=k)
    else:
        arr = xp.triu(x.arr, k=k)
        mask = xp.triu(mask, k=k)

    src_arr_ids = xp.where(mask, x.arr_ids, xp.full(1, -1, dtype=xp.int32))
    src_offsets = xp.where(mask, x.offsets, xp.full(1, -1, dtype=xp.int32))
    return Array(arr, src_arr_ids, src_offsets)


def tril(x, /, *, k=0):
    return _tri(x, lower=True, k=k)


def triu(x, /, *, k=0):
    return _tri(x, lower=False, k=k)


def zeros(shape, *, dtype=None, device=None):
    arr = nxp.zeros(shape, dtype=dtype, device=device)
    return Array(arr)


def zeros_like(x, /, *, dtype=None, device=None):
    arr = nxp.zeros_like(x.arr, dtype=dtype, device=device)
    return Array(arr, x.arr_ids, x.offsets)
