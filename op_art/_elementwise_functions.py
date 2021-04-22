# Element-wise Functions
# https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html

# These all follow the same pattern, according to whether the function is unary or binary.

import numpy as np

from ._array_object import _elementwise_unary_operation, _elementwise_binary_operation

def abs(x, /):
    return _elementwise_unary_operation(x, np.abs)

def acos(x, /):
    return _elementwise_unary_operation(x, np.arccos)

def acosh(x, /):
    return _elementwise_unary_operation(x, np.arccosh)

def add(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.add)

def asin(x, /):
    return _elementwise_unary_operation(x, np.arcsin)

def asinh(x, /):
    return _elementwise_unary_operation(x, np.arcsinh)

def atan(x, /):
    return _elementwise_unary_operation(x, np.arctan)

def atan2(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.arctan2)

def atanh(x, /):
    return _elementwise_unary_operation(x, np.arctanh)

def bitwise_and(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.bitwise_and)

def bitwise_invert(x, /):
    return _elementwise_unary_operation(x, np.invert)

def bitwise_left_shift(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.left_shift)

def bitwise_or(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.bitwise_or)

def bitwise_right_shift(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.right_shift)

def bitwise_xor(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.bitwise_xor)

def ceil(x, /):
    return _elementwise_unary_operation(x, np.ceil)

def cos(x, /):
    return _elementwise_unary_operation(x, np.cos)

def cosh(x, /):
    return _elementwise_unary_operation(x, np.cosh)

def divide(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.divide)

def equal(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.equal)

def exp(x, /):
    return _elementwise_unary_operation(x, np.exp)

def expm1(x, /):
    return _elementwise_unary_operation(x, np.expm1)

def floor(x, /):
    return _elementwise_unary_operation(x, np.floor)

def floor_divide(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.floor_divide)

def greater(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.greater)

def greater_equal(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.greater_equal)

def isfinite(x, /):
    return _elementwise_unary_operation(x, np.isfinite)

def isinf(x, /):
    return _elementwise_unary_operation(x, np.isinf)

def isnan(x, /):
    return _elementwise_unary_operation(x, np.isnan)

def less(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.less)

def less_equal(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.less_equal)

def log(x, /):
    return _elementwise_unary_operation(x, np.log)

def log1p(x, /):
    return _elementwise_unary_operation(x, np.log1p)

def log2(x, /):
    return _elementwise_unary_operation(x, np.log2)

def log10(x, /):
    return _elementwise_unary_operation(x, np.log10)

def logaddexp(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.logaddexp)

def logical_and(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.logical_and)

def logical_not(x, /):
    return _elementwise_unary_operation(x, np.logical_not)

def logical_or(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.logical_or)

def logical_xor(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.logical_xor)

def multiply(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.multiply)

def negative(x, /):
    return _elementwise_unary_operation(x, np.negative)

def not_equal(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.not_equal)

def positive(x, /):
    return _elementwise_unary_operation(x, np.positive)

def pow(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.power)

def remainder(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.remainder)

def round(x, /):
    return _elementwise_unary_operation(x, np.round)

def sign(x, /):
    return _elementwise_unary_operation(x, np.sign)

def sin(x, /):
    return _elementwise_unary_operation(x, np.sin)

def sinh(x, /):
    return _elementwise_unary_operation(x, np.sinh)

def square(x, /):
    return _elementwise_unary_operation(x, np.square)

def sqrt(x, /):
    return _elementwise_unary_operation(x, np.sqrt)

def subtract(x1, x2, /):
    return _elementwise_binary_operation(x1, x2, np.subtract)

def tan(x, /):
    return _elementwise_unary_operation(x, np.tan)

def tanh(x, /):
    return _elementwise_unary_operation(x, np.tanh)

def trunc(x, /):
    return _elementwise_unary_operation(x, np.trunc)
