# Statistical Functions
# https://data-apis.org/array-api/latest/API_specification/statistical_functions.html

# These all follow the same pattern.

import operator
import numpy as np

from ._array_object import _reduction_operation

def max(x, /, *, axis=None, keepdims=False):
    return _reduction_operation(x, axis, np.max)

def mean(x, /, *, axis=None, keepdims=False):
    # Don't do sanity check since mean isn't distributive
    return _reduction_operation(x, axis, np.mean)

def min(x, /, *, axis=None, keepdims=False):
    return _reduction_operation(x, axis, np.min)

def prod(x, /, *, axis=None, keepdims=False):
    return _reduction_operation(x, axis, np.prod)

def std(x, /, *, axis=None, correction=0.0, keepdims=False):
    # TODO: correction
    return _reduction_operation(x, axis, np.std)

def sum(x, /, *, axis=None, keepdims=False):
    return _reduction_operation(x, axis, np.sum, operator.add, 0)

def var(x, /, *, axis=None, correction=0.0, keepdims=False):
    # TODO: correction
    return _reduction_operation(x, axis, np.var)

# TODO: it would be nice if the animation for max highlighted the maximum values (use argmax to find)
