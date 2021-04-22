import builtins
import numpy as np

def index_to_offset(arr, index):
    return builtins.sum((np.array(index) * arr.strides).tolist())


def offset_to_index(arr, offset):
    return list(np.ndindex(*arr.shape))[offset // arr.itemsize]


# from https://numpy.org/doc/stable/reference/arrays.nditer.html#putting-the-inner-loop-in-cython
def axis_to_axeslist(axis, ndim):
    if axis is None:
        return [-1] * ndim
    else:
        if type(axis) is not tuple:
            axis = (axis,)
        axeslist = [1] * ndim
        for i in axis:
            axeslist[i] = -1
        ax = 0
        for i in range(ndim):
            if axeslist[i] != -1:
                axeslist[i] = ax
                ax += 1
        return axeslist