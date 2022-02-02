from dataclasses import asdict, dataclass
from typing import Any, Tuple
import builtins
import itertools
import json
import numpy as np
import numpy.array_api as nxp

from ._dtypes import _boolean_dtypes, _floating_dtypes

id_gen = itertools.count()
id_to_array = {}

# TODO: use a context manager or similar for this
def reset_ids():
    global id_gen
    global id_to_array
    id_gen = itertools.count()
    id_to_array = {}

class Array:
    def __init__(self, arr, src_arr_ids=None, src_offsets=None, *, id=None):
        if isinstance(arr, np.ndarray):
            raise ValueError("can't be ndarray")
        if isinstance(arr, np.generic):
            # Convert the array scalar to a 0-D array
            arr = np.asarray(arr)
        self.arr = arr
        self.id = id if id is not None else next(id_gen)
        self.arr_ids = nxp.broadcast_to(nxp.asarray(self.id, dtype=nxp.int32), self.arr.shape)
        self.offsets = nxp.arange(0, arr.size, dtype=nxp.int32)
        self.offsets = nxp.reshape(self.offsets, arr.shape)
        self.src_arr_ids = src_arr_ids
        self.src_offsets = src_offsets
        # ensure sources always have one more dimension than the array
        if self.src_arr_ids is not None and self.arr.ndim == self.src_arr_ids.ndim:
            self.src_arr_ids = nxp.expand_dims(self.src_arr_ids, axis=-1)
            self.src_offsets = nxp.expand_dims(self.src_offsets, axis=-1)
        self.representation = Array._get_representation(self.arr, self.id, self.offsets, src_arr_ids, src_offsets)
        assert self.representation.id == self.id
        id_to_array[self.id] = self
  
    @property
    def device(self):
        return self.arr.device

    @property
    def dtype(self):
        return self.arr.dtype

    @property
    def mT(self):
        from ._linear_algebra_functions import matrix_transpose
        return matrix_transpose(self)

    @property
    def ndim(self):
        return self.arr.ndim

    @property
    def shape(self):
        return self.arr.shape

    @property
    def size(self):
        return self.arr.size

    @property
    def T(self):
        if self.ndim != 2:
            raise ValueError("x.T requires x to have 2 dimensions. Use x.mT to transpose stacks of matrices and permute_dims() to permute dimensions.")
        arr = self.arr.T
        src_arr_ids = self.arr_ids.T
        src_offsets = self.offsets.T
        return Array(arr, src_arr_ids, src_offsets)


    def to_device(self, device, /, stream=None):
        return self.arr.to_device(device, stream)

    # Based on https://github.com/data-apis/numpy/blob/array-api/numpy/_array_api/_array_object.py
    # Helper function to match the type promotion rules in the spec
    def _promote_scalar(self, scalar):
        if isinstance(scalar, bool):
            if self.dtype not in _boolean_dtypes:
                raise TypeError("Python bool scalars can only be promoted with bool arrays")
        elif isinstance(scalar, int):
            if self.dtype in _boolean_dtypes:
                raise TypeError("Python int scalars cannot be promoted with bool arrays")
        elif isinstance(scalar, float):
            if self.dtype not in _floating_dtypes:
                raise TypeError("Python float scalars can only be promoted with floating-point arrays.")
        else:
            raise TypeError("'scalar' must be a Python scalar")

        return Array(nxp.asarray(scalar, dtype=self.dtype))

    def __getitem__(self, item):
        if isinstance(item, Array): # boolean array
            # TODO: this array should be a source too
            item = item.arr
        indexed_arr = self.arr[item]
        indexed_src_arr_ids = self.arr_ids[item]
        indexed_src_offsets = self.offsets[item]
        return Array(indexed_arr, indexed_src_arr_ids, indexed_src_offsets)

    def __setitem__(self, item, value):
        if isinstance(item, Array): # boolean array
            # TODO: this array should be a source too
            item = item.arr

        if isinstance(value, (int, float, bool)):
            value = self._promote_scalar(value)
        self.arr[item] = value.arr

        if self.src_arr_ids is None:
            self.src_arr_ids = np.full_like(self.arr, -1, dtype=np.int32)
            self.src_offsets = np.full_like(self.arr, -1, dtype=np.int32)

        if self.offsets.ndim == self.src_arr_ids.ndim:
            src_item = item
        else:
            src_item = item + (Ellipsis,)
        if value.arr_ids.shape != self.src_arr_ids[src_item].shape:
            # there are more sources in self than value, so expand value arrays to match
            value_arr_ids = np.full_like(self.src_arr_ids[src_item], -1, dtype=np.int32)
            value_offsets = np.full_like(self.src_offsets[src_item], -1, dtype=np.int32)
            value_arr_ids[..., 0] = value.arr_ids
            value_offsets[..., 0] = value.offsets
            self.src_arr_ids[src_item] = value_arr_ids
            self.src_offsets[src_item] = value_offsets
        else:
            self.src_arr_ids[src_item] = value.arr_ids
            self.src_offsets[src_item] = value.offsets
        self.representation = Array._get_representation(self.arr, self.id, self.offsets, self.src_arr_ids, self.src_offsets)

    def __abs__(self, /):
        return Array(self.arr.__abs__(), self.arr_ids, self.offsets)

    def __add__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        # Let the underlying arr lib do the calculation - and just do the
        # source updates in _elementwise_binary_operation2
        arr = self.arr.__add__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __and__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__and__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __array_namespace__(self, /, *, api_version=None):
        if api_version is not None and not api_version.startswith("2021."):
            raise ValueError("Unrecognized array API version")
        import op_art
        return op_art

    def __bool__(self, /):
        if self.shape != ():
            raise TypeError("bool is only allowed on arrays with shape ()")
        return self.arr.__bool__()

    def __dlpack__(self, /, *, stream=None):
        return self.arr.__dlpack__(stream=stream)

    def __dlpack_device__(self, /):
        return self.arr.__dlpack_device__()

    def __eq__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__eq__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __float__(self, /):
        if self.shape != ():
            raise TypeError("float is only allowed on arrays with shape ()")
        return self.arr.__float__()

    def __floordiv__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__floordiv__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __ge__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__ge__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __gt__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__gt__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __int__(self, /):
        if self.shape != ():
            raise TypeError("int is only allowed on arrays with shape ()")
        return self.arr.__int__()

    def __index__(self, /):
        return self.arr.__index__()

    def __invert__(self, /):
        return Array(self.arr.__invert__(), self.arr_ids, self.offsets)

    def __le__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__le__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __lshift__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__lshift__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __lt__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__lt__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __matmul__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        import op_art
        return op_art.matmul(self, other)

    def __mod__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__mod__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __mul__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__mul__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __ne__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__ne__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __neg__(self, /):
        return Array(self.arr.__neg__(), self.arr_ids, self.offsets)

    def __or__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__or__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __pos__(self, /):
        return Array(self.arr.__pos__(), self.arr_ids, self.offsets)

    def __pow__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__pow__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __rshift__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__rshift__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __sub__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__sub__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __truediv__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__truediv__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __xor__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__xor__(other.arr)
        return _elementwise_binary_operation2(self, other, arr)

    def __iadd__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        self.arr.__iadd__(other.arr)
        _inplace_elementwise_binary_operation2(self, other)
        return self

    def __radd__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__radd__(other.arr)
        return _elementwise_binary_operation2(other, self, arr)

    def __iand__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        self.arr.__iand__(other.arr)
        _inplace_elementwise_binary_operation2(self, other)
        return self

    def __rand__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__rand__(other.arr)
        return _elementwise_binary_operation2(other, self, arr)

    def __ifloordiv__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        self.arr.__ifloordiv__(other.arr)
        _inplace_elementwise_binary_operation2(self, other)
        return self

    def __rfloordiv__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__rfloordiv__(other.arr)
        return _elementwise_binary_operation2(other, self, arr)

    def __ilshift__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        self.arr.__ilshift__(other.arr)
        _inplace_elementwise_binary_operation2(self, other)
        return self

    def __rlshift__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__rlshift__(other.arr)
        return _elementwise_binary_operation2(other, self, arr)

    def __imatmul__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        self.arr.__imatmul__(other.arr)
        _inplace_elementwise_binary_operation2(self, other)
        return self

    def __rmatmul__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        import op_art
        return op_art.matmul(other, self)

    def __imod__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        self.arr.__imod__(other.arr)
        _inplace_elementwise_binary_operation2(self, other)
        return self

    def __rmod__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__rmod__(other.arr)
        return _elementwise_binary_operation2(other, self, arr)

    def __imul__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        self.arr.__imul__(other.arr)
        _inplace_elementwise_binary_operation2(self, other)
        return self

    def __rmul__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__rmul__(other.arr)
        return _elementwise_binary_operation2(other, self, arr)

    def __ior__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        self.arr.__ior__(other.arr)
        _inplace_elementwise_binary_operation2(self, other)
        return self

    def __ror__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__ror__(other.arr)
        return _elementwise_binary_operation2(other, self, arr)

    def __ipow__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        self.arr.__ipow__(other.arr)
        _inplace_elementwise_binary_operation2(self, other)
        return self

    def __rpow__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__rpow__(other.arr)
        return _elementwise_binary_operation2(other, self, arr)

    def __irshift__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        self.arr.__irshift__(other.arr)
        _inplace_elementwise_binary_operation2(self, other)
        return self

    def __rrshift__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__rrshift__(other.arr)
        return _elementwise_binary_operation2(other, self, arr)

    def __isub__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        self.arr.__isub__(other.arr)
        _inplace_elementwise_binary_operation2(self, other)
        return self

    def __rsub__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__rsub__(other.arr)
        return _elementwise_binary_operation2(other, self, arr)

    def __itruediv__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        self.arr.__itruediv__(other.arr)
        _inplace_elementwise_binary_operation2(self, other)
        return self

    def __rtruediv__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__rtruediv__(other.arr)
        return _elementwise_binary_operation2(other, self, arr)

    def __ixor__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        self.arr.__ixor__(other.arr)
        _inplace_elementwise_binary_operation2(self, other)
        return self

    def __rxor__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        arr = self.arr.__rxor__(other.arr)
        return _elementwise_binary_operation2(other, self, arr)


    @staticmethod
    def _get_representation(arr, id, offsets, src_arr_ids, src_offsets):
        """Convert array into a representation suitable for visualization"""
        cells = []
        it = np.nditer(arr, flags=["multi_index", "refs_ok", "zerosize_ok"], order="C")
        for x in it:
            ind = it.multi_index
            offset = offsets[ind]
            cell_id = f"{id}_{offset}"
            if src_arr_ids is None:
                cell_sources = None
            else:
                if offsets.ndim == src_arr_ids.ndim:
                    src_ind = ind
                else:
                    src_ind = ind + (Ellipsis,)
                src_arr_id = src_arr_ids[src_ind]
                if src_arr_id.ndim == 0:
                    if src_arr_id == -1:
                        cell_sources = None
                    else:
                        src_offset = src_offsets[src_ind]
                        cell_sources = [f"{src_arr_id}_{src_offset}"]
                else:
                    if nxp.all(src_arr_id == -1):
                        cell_sources = None
                    else:
                        src_offset = src_offsets[src_ind]
                        cell_sources = [f"{i}_{o}" for i, o in zip(src_arr_id, src_offset) if i != -1]
            cells.append(CellRepresentation(cell_id, it.multi_index, x.item(), cell_sources))
        return ArrayRepresentation(id, arr.dtype.kind, arr.ndim, arr.shape, tuple(cells))

    def to_json(self, visible_ids=None):
        from ._visualization import rewrite_representation
        return asdict(rewrite_representation(self.representation, visible_ids))

    def __repr__(self):
        return f"Array(id={self.id},\narr={self.arr},\nsrc_arr_ids={self.src_arr_ids},\nsrc_offsets={self.src_offsets},\nrepresentation={self.representation})"

    def _repr_javascript_(self):
        return """
        (function(element){
            require(['op_art'], function(op_art) {
                op_art.visualize(element.get(0), %s, [""], 1500, true);
            });
        })(element);
        """ % json.dumps([self.to_json(visible_ids=[self.id])])

@dataclass()
class CellRepresentation:
    id: Any
    index: Any
    value: Any
    sources: Any = None


@dataclass()
class ArrayRepresentation:
    id: int
    kind: str
    ndim: int
    shape: Any
    cells: Tuple[CellRepresentation]


def _elementwise_unary_operation(x, array_op):
    arr = array_op(x.arr)
    return Array(arr, x.arr_ids, x.offsets)

# copied from https://github.com/data-apis/numpy/blob/array-api/numpy/_array_api/_array_object.py
def _normalize_two_args(x1, x2):
    """
    Normalize inputs to two arg functions to fix type promotion rules
    NumPy deviates from the spec type promotion rules in cases where one
    argument is 0-dimensional and the other is not. For example:
    >>> import numpy as np
    >>> a = np.array([1.0], dtype=np.float32)
    >>> b = np.array(1.0, dtype=np.float64)
    >>> np.add(a, b) # The spec says this should be float64
    array([2.], dtype=float32)
    To fix this, we add a dimension to the 0-dimension array before passing it
    through. This works because a dimension would be added anyway from
    broadcasting, so the resulting shape is the same, but this prevents NumPy
    from not promoting the dtype.
    """
    if x1.shape == () and x2.shape != ():
        # The _array[None] workaround was chosen because it is relatively
        # performant. broadcast_to(x1._array, x2.shape) is much slower. We
        # could also manually type promote x2, but that is more complicated
        # and about the same performance as this.
        x1 = Array(x1.arr[None])
    elif x2.shape == () and x1.shape != ():
        x2 = Array(x2.arr[None])
    return (x1, x2)

def _elementwise_binary_operation(x1, x2, array_op):
    #x1, x2 = _normalize_two_args(x1, x2)
    arr = array_op(x1.arr, x2.arr)

    # broadcast if necessary
    from ._data_type_functions import broadcast_arrays
    x1_broad, x2_broad = broadcast_arrays(x1, x2)

    src_arr_ids = nxp.stack([x1_broad.arr_ids, x2_broad.arr_ids], axis=-1)
    src_offsets = nxp.stack([x1_broad.offsets, x2_broad.offsets], axis=-1)

    return Array(arr, src_arr_ids, src_offsets)

def _elementwise_binary_operation2(x1, x2, arr):
    x1_arr_ids, x2_arr_ids = nxp.broadcast_arrays(x1.arr_ids, x2.arr_ids)
    x1_offsets, x2_offsets = nxp.broadcast_arrays(x1.offsets, x2.offsets)

    src_arr_ids = nxp.stack([x1_arr_ids, x2_arr_ids], axis=-1)
    src_offsets = nxp.stack([x1_offsets, x2_offsets], axis=-1)

    return Array(arr, src_arr_ids, src_offsets)

def _inplace_elementwise_binary_operation2(x1, x2):
    x1_arr_ids, x2_arr_ids = nxp.broadcast_arrays(x1.arr_ids, x2.arr_ids)
    x1_offsets, x2_offsets = nxp.broadcast_arrays(x1.offsets, x2.offsets)

    x1.src_arr_ids = nxp.stack([x1_arr_ids, x2_arr_ids], axis=-1)
    x1.src_offsets = nxp.stack([x1_offsets, x2_offsets], axis=-1)

def _structural_operation(x, array_op, *args, **kwargs):
    # Suitable for operations that change the structure of x, not its values
    arr = array_op(x.arr, *args, **kwargs)
    src_arr_ids = array_op(x.arr_ids, *args, **kwargs)
    src_offsets = array_op(x.offsets, *args, **kwargs)
    return Array(arr, src_arr_ids, src_offsets)

def _reduction_operation(x, axis, array_op, keepdims=False, **kwargs):
    xp = x.arr.__array_namespace__()

    arr = array_op(x.arr, axis=axis, **kwargs)

    def normalize_axis(arr, a):
        # convert negative axis a to be >= 0
        return a if a >= 0 else a + arr.ndim

    # note that axis can be None, a single int, or a tuple of ints
    if x.ndim == 0:
        src_arr_ids = x.arr_ids
        src_offsets = x.offsets
    elif axis is None:
        src_arr_ids = xp.reshape(x.arr_ids, -1)
        src_offsets = xp.reshape(x.offsets, -1)
    elif isinstance(axis, int):
        # reduction in axis is equivalent to permuting the source arrays, so the
        # axis being reduced becomes the last one
        def move_axis_to_end(arr, axis):
            axis = normalize_axis(arr, axis)
            axes = list(range(arr.ndim))
            del axes[axis]
            axes.append(axis)
            return xp.permute_dims(arr, axes)
        src_arr_ids = move_axis_to_end(x.arr_ids, axis)
        src_offsets = move_axis_to_end(x.offsets, axis)
    else:
        # when axis specifies multiple axes, move them to the end, then flatten them
        def move_axis_to_end(arr, axis):
            axis = tuple(normalize_axis(arr, a) for a in axis)
            axes = list(range(arr.ndim))
            axes = [a for a in axes if a not in axis]
            axes.extend(axis)
            res = xp.permute_dims(arr, axes)

            shape = tuple([arr.shape[a] for a in axes if a not in axis]) + (-1,)
            return xp.reshape(res, shape)

        src_arr_ids = move_axis_to_end(x.arr_ids, axis)
        src_offsets = move_axis_to_end(x.offsets, axis)
        

    if keepdims and x.ndim > 0:
        if axis is None:
            axis = tuple(range(x.ndim))
        arr = xp.expand_dims(arr, axis=axis)
        src_arr_ids = xp.expand_dims(src_arr_ids, axis=axis)
        src_offsets = xp.expand_dims(src_offsets, axis=axis)

    return Array(arr, src_arr_ids, src_offsets)
