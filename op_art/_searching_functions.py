# Searching Functions
# https://data-apis.org/array-api/latest/API_specification/searching_functions.html

import numpy as np

from ._array_object import Array, _normalize_two_args, _reduction_operation

def argmax(x, /, *, axis=None, keepdims=False):
    return _reduction_operation(x, axis, np.argmax)

def argmin(x, /, *, axis=None, keepdims=False):
    return _reduction_operation(x, axis, np.argmin)

def nonzero(x, /):
    arrs = tuple(Array(i) for i in np.nonzero(x.arr))

    x_index_to_cell = {cell.index: cell for cell in x.representation.cells}
    # iterate over zipped cells as coordinates into original array (x)
    for cells in zip(*[arr.representation.cells for arr in arrs]):
        source_index = tuple(c.value for c in cells)
        source_id = x_index_to_cell[source_index].id
        for cell in cells:
            cell.sources = [source_id]
    return arrs

def where(condition, x1, x2, /):
    x1, x2 = _normalize_two_args(x1, x2)
    arr = np.where(condition.arr, x1.arr, x2.arr)
    rep = Array._get_representation(arr)

    # TODO: broadcast if necessary (like in _elementwise_binary_operation)

    condition_index_to_cell = {cell.index: cell for cell in condition.representation.cells}
    x1_index_to_cell = {cell.index: cell for cell in x1.representation.cells}
    x2_index_to_cell = {cell.index: cell for cell in x2.representation.cells}
    for cell in rep.cells:
        condition_cell = condition_index_to_cell[cell.index]
        x1_cell = x1_index_to_cell[cell.index]
        x2_cell = x2_index_to_cell[cell.index]
        if condition_cell.value:
            cell.sources = [condition_cell.id, x1_cell.id]
        else:
            cell.sources = [condition_cell.id, x2_cell.id]
    return Array.from_representation(arr, rep)