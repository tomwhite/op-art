# Set Functions
# https://data-apis.org/array-api/latest/API_specification/set_functions.html

import numpy as np

from typing import NamedTuple

from ._array_object import Array, _direct_mapping

# TODO: support >1D arrays

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

def _unique_from_inverse(x, arr, inv):
    rep = Array._get_representation(arr)
    for arr_cell, inv_index in zip(x.representation.cells, inv):
        cell = rep.cells[inv_index]
        sources = cell.sources or []
        sources.append(arr_cell.id)
        cell.sources = sources
    return Array.from_representation(arr, rep)

def unique_all(x, /):
    arr, ind, inv, counts = np.unique(x.arr, return_index=True, return_inverse=True, return_counts=True)

    values = _unique_from_inverse(x, arr, inv)
    indices = _unique_from_inverse(x, ind, inv)
    inverse_indices = _direct_mapping(x, inv.reshape(x.shape))
    counts = _unique_from_inverse(x, counts, inv)

    return UniqueAllResult(values, indices, inverse_indices, counts)

def unique_counts(x, /):
    arr, inv, counts = np.unique(x.arr, return_inverse=True, return_counts=True)

    values = _unique_from_inverse(x, arr, inv)
    counts = _unique_from_inverse(x, counts, inv)

    return UniqueCountsResult(values, counts)

def unique_inverse(x, /):
    arr, inv = np.unique(x.arr, return_inverse=True)

    values = _unique_from_inverse(x, arr, inv)
    inverse_indices = _direct_mapping(x, inv.reshape(x.shape))

    return UniqueInverseResult(values, inverse_indices)

def unique_values(x, /):
    arr, inv = np.unique(x.arr, return_inverse=True)

    values = _unique_from_inverse(x, arr, inv)

    return values
