# Linear Algebra Functions
# https://data-apis.org/array-api/latest/API_specification/linear_algebra_functions.html#

# Good animations are hard to implement for these functions!

from ._array_object import _structural_operation
from ._dtypes import _numeric_dtypes
from ._einsum import einsum

def matmul(x1, x2, /):
    xp = x1.arr.__array_namespace__()
    xp.matmul(x1.arr, x2.arr) # call for error checking
    if x1.ndim == 1 and x2.ndim == 1:
        return einsum("i,i->", x1, x2)
    elif x1.ndim == 1 and x2.ndim == 2:
        return einsum("i,ij->j", x1, x2)
    elif x1.ndim == 2 and x2.ndim == 1:
        return einsum("ij,j->i", x1, x2)
    else:
        return einsum("...ij,...jk->...ik", x1, x2)

def matrix_transpose(x, /):
    xp = x.arr.__array_namespace__()
    return _structural_operation(x, xp.matrix_transpose)

def tensordot(x1, x2, /, *, axes=2):
    xp = x1.arr.__array_namespace__()
    xp.tensordot(x1.arr, x2.arr, axes=axes) # call for error checking
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
