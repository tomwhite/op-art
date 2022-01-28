# Sorting Functions
# https://data-apis.org/array-api/latest/API_specification/sorting_functions.html

import numpy as np

from ._array_object import Array

def argsort(x, /, *, axis=-1, descending=False, stable=True):
    kind = "stable" if stable else "quicksort"
    arr = np.argsort(x.arr, axis, kind=kind)
    src_arr_ids = np.take_along_axis(x.arr_ids, arr, axis=axis)
    src_offsets = np.take_along_axis(x.offsets, arr, axis=axis)
    return Array(arr, src_arr_ids, src_offsets)

def sort(x, /, *, axis=-1, descending=False, stable=True):
    kind = "stable" if stable else "quicksort"
    arr = np.sort(x.arr, axis, kind=kind)
    ind = np.argsort(x.arr, axis, kind=kind) # use argsort to track inputs
    src_arr_ids = np.take_along_axis(x.arr_ids, ind, axis=axis)
    src_offsets = np.take_along_axis(x.offsets, ind, axis=axis)
    return Array(arr, src_arr_ids, src_offsets)
