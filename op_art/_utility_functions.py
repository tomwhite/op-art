# Utility Functions
# https://data-apis.org/array-api/latest/API_specification/utility_functions.html

import operator
import numpy as np

from ._array_object import _reduction_operation

def all(x, /, *, axis=None, keepdims=False):
    return _reduction_operation(x, axis, np.all)

def any(x, /, *, axis=None, keepdims=False):
    return _reduction_operation(x, axis, np.any)
