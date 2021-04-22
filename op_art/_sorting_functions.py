# Sorting Functions
# https://data-apis.org/array-api/latest/API_specification/sorting_functions.html

# TODO: argsort: like argmin

import numpy as np

from ._array_object import _index_mapping

def argsort(x, /, *, axis=-1, descending=False, stable=True):
    arr = np.argsort(x.arr, axis)
    ind = arr
    return _index_mapping(x, arr, ind)

def sort(x, /, *, axis=-1, descending=False, stable=True):
    arr = np.sort(x.arr, axis)
    ind = np.argsort(x.arr, axis) # use argsort to track inputs
    return _index_mapping(x, arr, ind)
