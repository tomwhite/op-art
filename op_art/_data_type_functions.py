# Data type functions
# https://data-apis.org/array-api/latest/API_specification/data_type_functions.html

import numpy as np

from ._array_object import Array, _broadcast_to

def broadcast_arrays(*arrays):
    # based on the numpy implementation
    shape = np.broadcast_shapes(*[a.arr.shape for a in arrays])

    if all(array.shape == shape for array in arrays):
        return arrays

    return [broadcast_to(array, shape) for array in arrays]

def broadcast_to(x, /, shape):
    return _broadcast_to(x, shape)

def can_cast(from_, to, /):
    if isinstance(from_, Array):
        from_ = from_.arr
    return np.can_cast(from_, to)

def finfo(type, /):
    return np.finfo(type)

def iinfo(type, /):
    return np.iinfo(type)

def result_type(*arrays_and_dtypes):
    return np.result_type(*(a.arr if isinstance(a, Array) else a for a in arrays_and_dtypes))
