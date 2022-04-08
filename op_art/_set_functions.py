# Set Functions
# https://data-apis.org/array-api/latest/API_specification/set_functions.html

from typing import NamedTuple

import numpy as np

from ._array_object import Array


class UniqueAllResult(NamedTuple):
    values: Array
    indices: Array
    inverse_indices: Array
    counts: Array


class UniqueCountsResult(NamedTuple):
    values: Array
    counts: Array


class UniqueInverseResult(NamedTuple):
    values: Array
    inverse_indices: Array


def _unique(x):
    # use np namespace since we need some functions that aren't in the array api (searchsorted, append)

    x_arr = np.asarray(x.arr)
    arr, ind, inv, counts = np.unique(
        x_arr, return_index=True, return_inverse=True, return_counts=True
    )

    # computing the sources is tricky, since each cell can depend on up to max_count
    # input values, and they are not easily transformed from any of the outputs of np.unique
    # so we have to loop through ranges of the sorted values

    max_count = np.max(counts)

    # initialise the source arrays
    src_arr_ids = np.full((arr.shape[0], max_count), -1, dtype=np.int32)
    src_offsets = np.full((arr.shape[0], max_count), -1, dtype=np.int32)

    # sort the original sources by sorted inv (ravel to handle case when x is >1d)
    inv_sorted_ind = np.argsort(inv)
    sorted_src_arr_ids = np.asarray(x.arr_ids).reshape(-1)[inv_sorted_ind]
    sorted_src_offsets = np.asarray(x.offsets).reshape(-1)[inv_sorted_ind]

    # find the ranges of the values that are identical
    x_arr_flat = x_arr.reshape(-1)
    ranges = np.searchsorted(x_arr_flat[inv_sorted_ind], arr)
    ranges = np.append(ranges, x_arr_flat.shape[0])
    for i in range(len(ranges) - 1):
        size = ranges[i + 1] - ranges[i]
        sl = slice(ranges[i], ranges[i + 1])
        src_arr_ids[i, :size] = sorted_src_arr_ids[sl]
        src_offsets[i, :size] = sorted_src_offsets[sl]

    # convert back to xp arrays
    xp = x.arr.__array_namespace__()
    arr = xp.asarray(arr)
    ind = xp.asarray(ind)
    inv = xp.asarray(inv.reshape(x.shape))  # needs to be same shape as x
    counts = xp.asarray(counts)
    src_arr_ids = xp.asarray(src_arr_ids)
    src_offsets = xp.asarray(src_offsets)

    values = Array(arr, src_arr_ids, src_offsets)
    indices = Array(ind, src_arr_ids, src_offsets)
    inverse_indices = Array(inv, x.arr_ids, x.offsets)
    counts = Array(counts, src_arr_ids, src_offsets)
    return values, indices, inverse_indices, counts


def unique_all(x, /):
    values, indices, inverse_indices, counts = _unique(x)
    return UniqueAllResult(values, indices, inverse_indices, counts)


def unique_counts(x, /):
    values, _, _, counts = _unique(x)
    return UniqueCountsResult(values, counts)


def unique_inverse(x, /):
    values, _, inverse_indices, _ = _unique(x)
    return UniqueInverseResult(values, inverse_indices)


def unique_values(x, /):
    values, _, _, _ = _unique(x)
    return values
