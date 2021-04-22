# Searching Functions
# https://data-apis.org/array-api/latest/API_specification/searching_functions.html

# TODO; nonzero is odd since it returns a tuple of arrays - hard to animate well
# TODO: where: various ways to animate - involves three arrays that mutually broadcast!

import operator
import numpy as np

from ._array_object import _reduction_operation

def argmax(x, /, *, axis=None, keepdims=False):
    return _reduction_operation(x, axis, np.argmax)

def argmin(x, /, *, axis=None, keepdims=False):
    return _reduction_operation(x, axis, np.argmin)
