# Manipulation Functions
# https://data-apis.org/array-api/latest/API_specification/manipulation_functions.html

# In general, the mapping from the input index to output index needs to be implemented
# from scratch.

import numpy as np

from ._array_object import _direct_mapping

def concat(arrays, /, *, axis=0):
    # call np version for error cases 
    arr = np.concatenate([a.arr for a in arrays], axis=axis)

    import op_art as xp
    ndim = len(arrays[0].shape)
    offsets = np.cumsum([0] + [a.shape[axis] for a in arrays])
    result = xp.empty(arr.shape, dtype=arr.dtype)
    for i, a in enumerate(arrays):
        index = [slice(None)] * ndim
        index[axis] = slice(offsets[i], offsets[i + 1])
        index = tuple(index)
        result[index] = a
    return result

def expand_dims(x, /, *, axis):
    arr = np.expand_dims(x.arr, axis=axis)
    return _direct_mapping(x, arr)

def flip(x, /, *, axis=None):
    # we can just delegate to numpy, since it implements flip using indexing
    # see notes at https://numpy.org/doc/stable/reference/generated/numpy.flip.html
    return np.flip(x, axis)

def reshape(x, /, shape):
    arr = np.reshape(x.arr, shape)
    # TODO: reshape may do a copy (e.g. after flip or transpose), so we can't use _direct_mapping
    return _direct_mapping(x, arr)

def roll(x, /, shift, *, axis=None):
    # based on https://github.com/numpy/numpy/blob/v1.20.0/numpy/core/numeric.py#L1146-L1244
    import itertools
    from numpy.core.numeric import normalize_axis_tuple
    import op_art as xp
    a = x
    if axis is None:
        b = xp.reshape(a, -1) # ravel
        b = roll(b, shift, axis=0)
        return xp.reshape(b, a.shape)

    else:
        axis = normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        broadcasted = np.broadcast(shift, axis)
        if broadcasted.ndim > 1:
            raise ValueError(
                "'shift' and 'axis' should be scalars or 1D sequences")
        shifts = {ax: 0 for ax in range(a.ndim)}
        for sh, ax in broadcasted:
            shifts[ax] += sh

        rolls = [((slice(None), slice(None)),)] * a.ndim
        for ax, offset in shifts.items():
            offset %= a.shape[ax] or 1  # If `a` is empty, nothing matters.
            if offset:
                # (original, result), (original, result)
                rolls[ax] = ((slice(None, -offset), slice(offset, None)),
                             (slice(-offset, None), slice(None, offset)))

        result = xp.empty_like(a)
        for indices in itertools.product(*rolls):
            arr_index, res_index = zip(*indices)
            result[res_index] = a[arr_index]

        return result

def squeeze(x, /, axis):
    arr = np.squeeze(x.arr, axis=axis)
    return _direct_mapping(x, arr)

def stack(arrays, /, *, axis=0):
    # based on https://github.com/numpy/numpy/blob/v1.20.0/numpy/core/shape_base.py#L358-L434
    from numpy.core.numeric import normalize_axis_index
    import op_art as xp

    result_ndim = arrays[0].ndim + 1
    axis = normalize_axis_index(axis, result_ndim)

    expanded_arrays = [expand_dims(arr, axis=axis) for arr in arrays]
    return xp.concat(expanded_arrays, axis=axis)
