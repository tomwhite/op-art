# Statistical Functions
# https://data-apis.org/array-api/latest/API_specification/statistical_functions.html

from ._array_object import _reduction_operation


def max(x, /, *, axis=None, keepdims=False):
    xp = x.arr.__array_namespace__()
    return _reduction_operation(x, axis, xp.max, keepdims=keepdims)


def mean(x, /, *, axis=None, keepdims=False):
    xp = x.arr.__array_namespace__()
    return _reduction_operation(x, axis, xp.mean, keepdims=keepdims)


def min(x, /, *, axis=None, keepdims=False):
    xp = x.arr.__array_namespace__()
    return _reduction_operation(x, axis, xp.min, keepdims=keepdims)


def prod(x, /, *, axis=None, dtype=None, keepdims=False):
    xp = x.arr.__array_namespace__()
    return _reduction_operation(x, axis, xp.prod, keepdims=keepdims, dtype=dtype)


def std(x, /, *, axis=None, correction=0.0, keepdims=False):
    xp = x.arr.__array_namespace__()
    return _reduction_operation(
        x, axis, xp.std, keepdims=keepdims, correction=correction
    )


def sum(x, /, *, axis=None, dtype=None, keepdims=False):
    xp = x.arr.__array_namespace__()
    return _reduction_operation(x, axis, xp.sum, keepdims=keepdims, dtype=dtype)


def var(x, /, *, axis=None, correction=0.0, keepdims=False):
    xp = x.arr.__array_namespace__()
    return _reduction_operation(
        x, axis, xp.var, keepdims=keepdims, correction=correction
    )
