from dataclasses import asdict, dataclass
from typing import Any, Tuple
import builtins
import itertools
import json
import numpy as np

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
        if isinstance(arr, np.generic):
            # Convert the array scalar to a 0-D array
            arr = np.asarray(arr)
        self.arr = arr
        self.id = id if id is not None else next(id_gen)
        self.arr_ids = np.full_like(arr, self.id, dtype=np.int32)
        self.offsets = np.arange(0, arr.size * arr.itemsize, arr.itemsize, dtype=np.int32)\
            .reshape(arr.shape)
        self.src_arr_ids = src_arr_ids
        self.src_offsets = src_offsets
        self.representation = Array._get_representation(self.arr, self.id, self.offsets, src_arr_ids, src_offsets)
        assert self.representation.id == self.id
        id_to_array[self.id] = self
  
    @property
    def device(self):
        return "cpu"

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
        if stream is not None:
            raise ValueError("The stream argument to to_device() is not supported")
        if device == "cpu":
            return self
        raise ValueError(f"Unsupported device {device!r}")

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

        return Array(np.array(scalar, self.dtype))

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

        if value.arr_ids.shape != self.src_arr_ids[item].shape:
            # there are more sources in self than value, so expand value arrays to match
            value_arr_ids = np.full_like(self.src_arr_ids[item], -1, dtype=np.int32)
            value_offsets = np.full_like(self.src_offsets[item], -1, dtype=np.int32)
            value_arr_ids[..., 0] = value.arr_ids
            value_offsets[..., 0] = value.offsets
            self.src_arr_ids[item] = value_arr_ids
            self.src_offsets[item] = value_offsets
        else:
            self.src_arr_ids[item] = value.arr_ids
            self.src_offsets[item] = value.offsets
        self.representation = Array._get_representation(self.arr, self.id, self.offsets, self.src_arr_ids, self.src_offsets)

    def __abs__(self, /):
        return _elementwise_unary_operation(self, np.abs)

    def __add__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.add)

    def __and__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.logical_and)

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

        return _elementwise_binary_operation(self, other, np.equal)

    def __float__(self, /):
        if self.shape != ():
            raise TypeError("float is only allowed on arrays with shape ()")
        return self.arr.__float__()

    def __floordiv__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.floor_divide)

    def __ge__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.greater_equal)

    def __gt__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.greater)

    def __int__(self, /):
        if self.shape != ():
            raise TypeError("int is only allowed on arrays with shape ()")
        return self.arr.__int__()

    def __index__(self, /):
        return self.arr.__index__()

    def __invert__(self, /):
        return _elementwise_unary_operation(self, np.invert)

    def __le__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.less_equal)

    def __lshift__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.left_shift)

    def __lt__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.less)

    def __matmul__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        import op_art
        return op_art.matmul(self, other)

    def __mod__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.floor_divide)

    def __mul__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.multiply)

    def __ne__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.not_equal)

    def __neg__(self, /):
        return _elementwise_unary_operation(self, np.negative)

    def __or__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.logical_or)

    def __pos__(self, /):
        return _elementwise_unary_operation(self, np.positive)

    def __pow__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.power)

    def __rshift__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.right_shift)

    def __sub__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.subtract)

    def __truediv__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.divide)

    def __xor__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.logical_xor)

    # TODO: __iadd__

    def __radd__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(other, self, np.add)

    # TODO: __iand__

    def __rand__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(other, self, np.logical_and)

    # TODO: __ifloordiv__

    def __rfloordiv__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(other, self, np.floor_divide)

    # TODO: __ilshift__

    def __rlshift__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(other, self, np.left_shift)

    # TODO: __imatmul__

    def __rmatmul__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        import op_art
        return op_art.matmul(other, self)

    # TODO: __imod__

    def __rmod__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(other, self, np.floor_divide)

    # TODO: __imul__

    def __rmul__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(other, self, np.multiply)

    # TODO: __ior__

    def __ror__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(other, self, np.logical_or)

    # TODO: __ipow__

    def __rpow__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(other, self, np.power)

    # TODO: __irshift__

    def __rrshift__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(other, self, np.right_shift)

    # TODO: __isub__

    def __rsub__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(other, self, np.subtract)

    # TODO: __itruediv__

    def __rtruediv__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(other, self, np.divide)

    # TODO: __ixor__

    def __rxor__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(other, self, np.logical_xor)

    @staticmethod
    def _get_representation(arr, id, offsets, src_arr_ids, src_offsets):
        """Convert array into a representation suitable for visualization"""
        cells = []
        it = np.nditer(arr, flags=["multi_index", "refs_ok", "zerosize_ok"], order="C")
        for x in it:
            offset = offsets[it.multi_index]
            cell_id = f"{id}_{offset}"
            if src_arr_ids is None:
                cell_sources = None
            else:
                src_arr_id = src_arr_ids[it.multi_index]
                if np.isscalar(src_arr_id):
                    if src_arr_id == -1:
                        cell_sources = None
                    else:
                        src_offset = src_offsets[it.multi_index]
                        cell_sources = [f"{src_arr_id}_{src_offset}"]
                else:
                    if np.all(src_arr_id == -1):
                        cell_sources = None
                    else:
                        src_offset = src_offsets[it.multi_index]
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
    x1, x2 = _normalize_two_args(x1, x2)
    arr = array_op(x1.arr, x2.arr)

    # broadcast if necessary
    from ._data_type_functions import broadcast_arrays
    x1_broad, x2_broad = broadcast_arrays(x1, x2)

    src_arr_ids = np.stack([x1_broad.arr_ids, x2_broad.arr_ids], axis=-1)
    src_offsets = np.stack([x1_broad.offsets, x2_broad.offsets], axis=-1)

    return Array(arr, src_arr_ids, src_offsets)

def _structural_operation(x, array_op, *args, **kwargs):
    # Suitable for operations that change the structure of x, not its values
    arr = array_op(x.arr, *args, **kwargs)
    src_arr_ids = array_op(x.arr_ids, *args, **kwargs)
    src_offsets = array_op(x.offsets, *args, **kwargs)
    return Array(arr, src_arr_ids, src_offsets)

def _reduction_operation(x, axis, array_op, keepdims=False, **kwargs):
    arr = array_op(x.arr, axis, **kwargs)

    def normalize_axis(arr, a):
        # convert negative axis a to be >= 0
        return a if a >= 0 else a + arr.ndim

    # note that axis can be None, a single int, or a tuple of ints
    if axis is None:
        src_arr_ids = x.arr_ids.ravel()
        src_offsets = x.offsets.ravel()
    elif isinstance(axis, int):
        # reduction in axis is equivalent to permuting the source arrays, so the
        # axis being reduced becomes the last one
        def move_axis_to_end(arr, axis):
            axis = normalize_axis(arr, axis)
            axes = list(range(arr.ndim))
            del axes[axis]
            axes.append(axis)
            return np.transpose(arr, axes)
        src_arr_ids = move_axis_to_end(x.arr_ids, axis)
        src_offsets = move_axis_to_end(x.offsets, axis)
    else:
        # when axis specifies multiple axes, move them to the end, then flatten them
        def move_axis_to_end(arr, axis):
            axis = tuple(normalize_axis(arr, a) for a in axis)
            axes = list(range(arr.ndim))
            axes = [a for a in axes if a not in axis]
            axes.extend(axis)
            res = np.transpose(arr, axes)

            shape = tuple([arr.shape[a] for a in axes if a not in axis]) + (-1,)
            return res.reshape(shape)

        src_arr_ids = move_axis_to_end(x.arr_ids, axis)
        src_offsets = move_axis_to_end(x.offsets, axis)
        

    if keepdims and x.ndim > 0:
        if axis is None:
            axis = tuple(range(x.ndim))
        arr = np.expand_dims(arr, axis)
        src_arr_ids = np.expand_dims(src_arr_ids, axis)
        src_offsets = np.expand_dims(src_offsets, axis)

    return Array(arr, src_arr_ids, src_offsets)
