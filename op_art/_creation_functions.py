# Creation Functions
# https://data-apis.org/array-api/latest/API_specification/creation_functions.html

# Note: these can all delegate directly to the equivalent underlying array function

import numpy as np

from ._array_object import Array, _direct_mapping

# device is not supported

def arange(start, /, stop=None, step=1, *, dtype=None, device=None):
    arr = np.arange(start, stop, step, dtype)
    return Array(arr)

def asarray(obj, /, *, dtype=None, device=None, copy=None):
    # TODO: support copy
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

def full(shape, fill_value, *, dtype=None, device=None):
    arr = np.full(shape, fill_value, dtype)
    return Array(arr)

def full_like(x, /, fill_value, *, dtype=None, device=None):
    arr = np.full_like(x.arr, fill_value, dtype)
    return _direct_mapping(x, arr)

def linspace(start, stop, /, num, *, dtype=None, device=None, endpoint=True):
    arr = np.linspace(start, stop, num, dtype=dtype, endpoint=endpoint)
    return Array(arr)

def ones(shape, *, dtype=None, device=None):
    arr = np.ones(shape, dtype)
    return Array(arr)

def ones_like(x, /, *, dtype=None, device=None):
    arr = np.ones_like(x.arr, dtype)
    return _direct_mapping(x, arr)

def zeros(shape, *, dtype=None, device=None):
    arr = np.zeros(shape, dtype)
    return Array(arr)

def zeros_like(x, /, *, dtype=None, device=None):
    arr = np.zeros_like(x.arr, dtype)
    return _direct_mapping(x, arr)