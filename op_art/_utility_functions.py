# Utility Functions
# https://data-apis.org/array-api/latest/API_specification/utility_functions.html

from ._array_object import _reduction_operation

def all(x, /, *, axis=None, keepdims=False):
    xp = x.arr.__array_namespace__()
    return _reduction_operation(x, axis, xp.all, keepdims=keepdims)

def any(x, /, *, axis=None, keepdims=False):
    xp = x.arr.__array_namespace__()
    return _reduction_operation(x, axis, xp.any, keepdims=keepdims)
