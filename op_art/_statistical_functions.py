# Statistical Functions
# https://data-apis.org/array-api/latest/API_specification/statistical_functions.html

# These all follow the same pattern.

import operator
import numpy as np

from ._array_object import _reduction_operation
from ._dtypes import float32, float64

def max(x, /, *, axis=None, keepdims=False):
    return _reduction_operation(x, axis, np.max)

def mean(x, /, *, axis=None, keepdims=False):
    # Don't do sanity check since mean isn't distributive
    return _reduction_operation(x, axis, np.mean)

def min(x, /, *, axis=None, keepdims=False):
    return _reduction_operation(x, axis, np.min)

def prod(x, /, *, axis=None, dtype=None, keepdims=False):
    if dtype is None and x.dtype == float32:
        dtype = float64
    return _reduction_operation(x, axis, np.prod, dtype=dtype)

def std(x, /, *, axis=None, correction=0.0, keepdims=False):
    return _reduction_operation(x, axis, np.std, ddof=correction)

def sum(x, /, *, axis=None, dtype=None, keepdims=False):
    if dtype is None and x.dtype == float32:
        dtype = float64
    return _reduction_operation(x, axis, np.sum, py_op=operator.add, fill_value=0, dtype=dtype)

def var(x, /, *, axis=None, correction=0.0, keepdims=False):
    return _reduction_operation(x, axis, np.var, ddof=correction)

# TODO: it would be nice if the animation for max highlighted the maximum values (use argmax to find)
