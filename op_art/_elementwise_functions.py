# Element-wise Functions
# https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html

# These all follow the same pattern, according to whether the function is unary or binary.

import numpy as np
import numpy.array_api as nxp

from ._array_object import _elementwise_unary_operation, _elementwise_binary_operation
from ._dtypes import _integer_dtypes

def abs(x, /):
    return _elementwise_unary_operation(x, nxp.abs)

def acos(x, /):
    return _elementwise_unary_operation(x, nxp.acos)

def acosh(x, /):
    return _elementwise_unary_operation(x, nxp.acosh)

def add(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.add)

def asin(x, /):
    return _elementwise_unary_operation(x, nxp.asin)

def asinh(x, /):
    return _elementwise_unary_operation(x, nxp.asinh)

def atan(x, /):
    return _elementwise_unary_operation(x, nxp.atan)

def atan2(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.atan2)

def atanh(x, /):
    return _elementwise_unary_operation(x, nxp.atanh)

def bitwise_and(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.bitwise_and)

def bitwise_invert(x, /):
    return _elementwise_unary_operation(x, nxp.bitwise_invert)

def bitwise_left_shift(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.bitwise_left_shift)

def bitwise_or(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.bitwise_or)

def bitwise_right_shift(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.bitwise_right_shift)

def bitwise_xor(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.bitwise_xor)

def ceil(x, /):
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of ceil is the same as the input
        return x
    return _elementwise_unary_operation(x, nxp.ceil)

def cos(x, /):
    return _elementwise_unary_operation(x, nxp.cos)

def cosh(x, /):
    return _elementwise_unary_operation(x, nxp.cosh)

def divide(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.divide)

def equal(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.equal)

def exp(x, /):
    return _elementwise_unary_operation(x, nxp.exp)

def expm1(x, /):
    return _elementwise_unary_operation(x, nxp.expm1)

def floor(x, /):
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of floor is the same as the input
        return x
    return _elementwise_unary_operation(x, nxp.floor)

def floor_divide(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.floor_divide)

def greater(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.greater)

def greater_equal(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.greater_equal)

def isfinite(x, /):
    return _elementwise_unary_operation(x, nxp.isfinite)

def isinf(x, /):
    return _elementwise_unary_operation(x, nxp.isinf)

def isnan(x, /):
    return _elementwise_unary_operation(x, nxp.isnan)

def less(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.less)

def less_equal(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.less_equal)

def log(x, /):
    return _elementwise_unary_operation(x, nxp.log)

def log1p(x, /):
    return _elementwise_unary_operation(x, nxp.log1p)

def log2(x, /):
    return _elementwise_unary_operation(x, nxp.log2)

def log10(x, /):
    return _elementwise_unary_operation(x, nxp.log10)

def logaddexp(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.logaddexp)

def logical_and(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.logical_and)

def logical_not(x, /):
    return _elementwise_unary_operation(x, nxp.logical_not)

def logical_or(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.logical_or)

def logical_xor(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.logical_xor)

def multiply(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.multiply)

def negative(x, /):
    return _elementwise_unary_operation(x, nxp.negative)

def not_equal(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.not_equal)

def positive(x, /):
    return _elementwise_unary_operation(x, nxp.positive)

def pow(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.pow)

def remainder(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.remainder)

def round(x, /):
    return _elementwise_unary_operation(x, nxp.round)

def sign(x, /):
    return _elementwise_unary_operation(x, nxp.sign)

def sin(x, /):
    return _elementwise_unary_operation(x, nxp.sin)

def sinh(x, /):
    return _elementwise_unary_operation(x, nxp.sinh)

def square(x, /):
    return _elementwise_unary_operation(x, nxp.square)

def sqrt(x, /):
    return _elementwise_unary_operation(x, nxp.sqrt)

def subtract(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, nxp.subtract)

def tan(x, /):
    return _elementwise_unary_operation(x, nxp.tan)

def tanh(x, /):
    return _elementwise_unary_operation(x, nxp.tanh)

def trunc(x, /):
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of trunc is the same as the input
        return x
    return _elementwise_unary_operation(x, nxp.trunc)
