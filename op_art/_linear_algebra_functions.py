# Linear Algebra Functions
# https://data-apis.org/array-api/latest/API_specification/linear_algebra_functions.html#

# Good animations are hard to implement for these functions!

from ._array_object import _structural_operation
from ._dtypes import _numeric_dtypes
from ._einsum import einsum

def matmul(x1, x2, /):
    if x1.ndim != 2 or x2.ndim != 2:
        raise NotImplementedError("matmul only implemented for 2x2 arrays")

    return einsum("...ij,...jk->...ik", x1, x2)

def matrix_transpose(x, /):
    xp = x.arr.__array_namespace__()
    return _structural_operation(x, xp.matrix_transpose)

def tensordot(x1, x2, /, *, axes=2):
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in tensordot')
    if x1.ndim == 0:
        return x1
    elif x2.ndim == 0:
        return x2
    if isinstance(axes, int):
        axes = (range(-1, -1 - axes, -1), range(axes))
    # inspired by https://scicomp.stackexchange.com/a/34720
    x1_indexes = list(range(x1.ndim))
    x2_indexes = list(range(x1.ndim, x1.ndim + x2.ndim))
    for x1_ind, x2_ind in zip(*axes):
        x2_indexes[x2_ind] = x1_indexes[x1_ind]
    return einsum(x1, x1_indexes, x2, x2_indexes)

def vecdot(x1, x2, /, *, axis=-1):
    return tensordot(x1, x2, axes=((axis,), (axis,)))
