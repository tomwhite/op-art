# Sorting Functions
# https://data-apis.org/array-api/latest/API_specification/sorting_functions.html

import numpy as np

from ._array_object import Array

def argsort(x, /, *, axis=-1, descending=False, stable=True):
    xp = x.arr.__array_namespace__()
    arr = xp.argsort(x.arr, axis=axis, descending=descending, stable=stable)
    # convert back to numpy arrays to use np.take_along_axis
    src_arr_ids = np.take_along_axis(np.asarray(x.arr_ids), arr, axis=axis)
    src_offsets = np.take_along_axis(np.asarray(x.offsets), arr, axis=axis)
    return Array(arr, xp.asarray(src_arr_ids), xp.asarray(src_offsets))

def sort(x, /, *, axis=-1, descending=False, stable=True):
    xp = x.arr.__array_namespace__()
    arr = xp.sort(x.arr, axis=axis, descending=descending, stable=stable)
    ind = xp.argsort(x.arr, axis=axis, descending=descending, stable=stable) # use argsort to track inputs
    # convert back to numpy arrays to use np.take_along_axis
    src_arr_ids = np.take_along_axis(np.asarray(x.arr_ids), ind, axis=axis)
    src_offsets = np.take_along_axis(np.asarray(x.offsets), ind, axis=axis)
    return Array(arr, xp.asarray(src_arr_ids), xp.asarray(src_offsets))
