# Sorting Functions
# https://data-apis.org/array-api/latest/API_specification/sorting_functions.html

import numpy as np

from ._array_object import _index_mapping

def argsort(x, /, *, axis=-1, descending=False, stable=True):
    # For ndim>1 ind is index along axis, so _index_mapping won't work
    if x.ndim > 1:
        raise NotImplementedError("argsort not implemented for ndim larger than 1")
    kind = "stable" if stable else "quicksort"
    arr = np.argsort(x.arr, axis, kind=kind)
    ind = arr
    return _index_mapping(x, arr, ind)

def sort(x, /, *, axis=-1, descending=False, stable=True):
    # For ndim>1 ind is index along axis, so _index_mapping won't work
    if x.ndim > 1:
        raise NotImplementedError("sort not implemented for ndim larger than 1")
    kind = "stable" if stable else "quicksort"
    arr = np.sort(x.arr, axis, kind=kind)
    ind = np.argsort(x.arr, axis, kind=kind) # use argsort to track inputs
    return _index_mapping(x, arr, ind)
