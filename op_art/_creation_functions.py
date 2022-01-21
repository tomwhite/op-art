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
        xp.reshape(output[0], (1, -1) + s0[2:])
        xp.reshape(output[1], (-1, 1) + s0[2:])

    return xp.broadcast_arrays(*output)

def ones(shape, *, dtype=None, device=None):
    arr = np.ones(shape, dtype)
    return Array(arr)

def ones_like(x, /, *, dtype=None, device=None):
    arr = np.ones_like(x.arr, dtype)
    return _direct_mapping(x, arr)

def tril(x, /, *, k=0):
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for tril")
    if x.ndim > 2:
        raise NotImplementedError("tril only implemented for 2-dimensional arrays")

    arr = np.tril(x.arr, k=k)
    rep = Array._get_representation(arr)

    tril_indices = np.tril_indices_from(x.arr, k=k)
    tril_indices = list(zip(*tril_indices))

    x_index_to_cell = {cell.index: cell for cell in x.representation.cells}
    for cell in rep.cells:
        if cell.index in tril_indices:
            source_id = x_index_to_cell[cell.index].id
            cell.sources = [source_id]
    return Array.from_representation(arr, rep)

def triu(x, /, *, k=0):
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for triu")
    if x.ndim > 2:
        raise NotImplementedError("triu only implemented for 2-dimensional arrays")

    arr = np.triu(x.arr, k=k)
    rep = Array._get_representation(arr)

    triu_indices = np.triu_indices_from(x.arr, k=k)
    triu_indices = list(zip(*triu_indices))

    x_index_to_cell = {cell.index: cell for cell in x.representation.cells}
    for cell in rep.cells:
        if cell.index in triu_indices:
            source_id = x_index_to_cell[cell.index].id
            cell.sources = [source_id]
    return Array.from_representation(arr, rep)

def zeros(shape, *, dtype=None, device=None):
    arr = np.zeros(shape, dtype)
    return Array(arr)

def zeros_like(x, /, *, dtype=None, device=None):
    arr = np.zeros_like(x.arr, dtype)
    return _direct_mapping(x, arr)