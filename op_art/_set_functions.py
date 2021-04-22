# Set Functions
# https://data-apis.org/array-api/latest/API_specification/set_functions.html

import numpy as np

from ._array_object import _index_mapping

def unique(x, /, *, return_counts=False, return_index=False, return_inverse=False):
    # TODO: honour return_* args
    # note that numpy sorts, but this is optional in the array API spec
    arr, ind = np.unique(x.arr, return_index=True)
    return _index_mapping(x, arr, ind)