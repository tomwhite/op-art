from dataclasses import asdict, dataclass
from typing import Any, List, Tuple
import builtins
import collections
import inspect
import itertools
import json
import numpy as np
import operator
import os
from textwrap import dedent

from numpy.testing import assert_equal, assert_array_equal

from ._dtypes import _boolean_dtypes, _integer_dtypes, _floating_dtypes
from ._utils import axis_to_axeslist, index_to_offset, offset_to_index

id_gen = itertools.count()
id_to_array = {}

# TODO: use a context manager or similar for this
def reset_ids():
    global id_gen
    global id_to_array
    id_gen = itertools.count()
    id_to_array = {}

class Array:
    def __init__(self, arr, *, id=None):
        self.arr = arr
        self.id = id or next(id_gen)
        self.representation = Array._get_representation(self.arr, self.id)
        assert self.representation.id == self.id
        id_to_array[self.id] = self
  
    @property
    def device(self):
        raise NotImplementedError("The device attribute is not yet implemented")

    @property
    def dtype(self):
        return self.arr.dtype

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
        arr = self.arr.T
        # transpose doesn't change the underlying array
        return _direct_mapping(self, arr)

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
            item = item.arr
            args = np.argwhere(item)
            cell_indexes = [tuple(row) for row in args]
            # TODO: show item array as a source too

        else:
            # get indexes for selected cells
            grid = np.mgrid[tuple(slice(0, n) for n in self.arr.shape)]
            if not isinstance(item, tuple):
                item = (item, )
            s = (slice(0, len(grid)), ) + item
            grid_s = grid[s]
            ndim = self.ndim
            if ndim == 0:
                cell_indexes = []
            else:
                cell_indexes = [tuple(r) for r in grid_s.reshape(ndim, grid_s.size // ndim).T]

        indexed_arr = self.arr[item]
        rep = self.representation
        indexed_cells = [cell for cell in rep.cells if cell.index in cell_indexes]
        new_rep = Array._get_representation(indexed_arr)
        
        cells = []
        # this assumes that the indexing linearizes in the same order
        for cell1, cell2 in zip(indexed_cells, new_rep.cells):
            assert_equal(cell1.value, cell2.value)
            cell = CellRepresentation(cell2.id, cell2.index, cell2.value, [cell1.id])
            cells.append(cell)

        rep = ArrayRepresentation(new_rep.id, indexed_arr.dtype.kind, indexed_arr.ndim, indexed_arr.shape, cells)
        return Array.from_representation(indexed_arr, rep)

    def __setitem__(self, item, value):
        # TODO: remove duplication from __getitem__
        if isinstance(item, Array): # boolean array
            item = item.arr
            args = np.argwhere(item)
            cell_indexes = [tuple(row) for row in args]
            # TODO: show item array as a source too

        else:
            # get indexes for selected cells
            grid = np.mgrid[tuple(slice(0, n) for n in self.arr.shape)]
            if not isinstance(item, tuple):
                item = (item, )
            s = (slice(0, len(grid)), ) + item
            grid_s = grid[s]
            ndim = self.ndim
            if ndim == 0:
                cell_indexes = []
            else:
                cell_indexes = [tuple(r) for r in grid_s.reshape(ndim, grid_s.size // ndim).T]

        if isinstance(value, (int, float, bool)):
            value = self._promote_scalar(value)
        self.arr[item] = value.arr

        # this assumes that the indexing linearizes in the same order
        # TODO: don't mutate the representation - have a graph of representations, and add to it
        for index, new_cell in zip(cell_indexes, value.representation.cells):
            for cell in self.representation.cells:
                if cell.index == index:
                    cell.value = new_cell.value
                    cell.sources = [new_cell.id]

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
        if api_version is not None:
            raise ValueError("Unrecognized array API version")
        import op_art
        return op_art

    def __bool__(self, /):
        if self.shape != ():
            raise TypeError("bool is only allowed on arrays with shape ()")
        return self.arr.__bool__()

    # TODO: __dlpack__
    # TODO: __dlpack_device__

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

    def __invert__(self, /):
        return _elementwise_unary_operation(self, np.invert)

    def __le__(self, other, /):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        return _elementwise_binary_operation(self, other, np.less_equal)

    # TODO: __len__

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
    # TODO: __rmatmul__

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
    def _get_representation(arr, arr_id=None):
        """Convert array into a representation suitable for visualization"""
        if arr.ndim >= 4:
            raise NotImplementedError("Arrays of dimension 4 or more are not supported.")

        if arr_id is None:
            arr_id = next(id_gen)
        cells = []
        it = np.nditer(arr, flags=["multi_index", "zerosize_ok"])
        for x in it:
            offset = builtins.sum((np.array(it.multi_index) * arr.strides).tolist())  # tolist to convert to python int
            cell_id = f"{arr_id}_{offset}"
            cells.append(CellRepresentation(cell_id, it.multi_index, x.item()))
        return ArrayRepresentation(arr_id, arr.dtype.kind, arr.ndim, arr.shape, tuple(cells))

    @staticmethod
    def from_representation(arr, rep):
        """Create an array directly from cell dependencies (not input or output representations)"""
        a = Array(arr, id=rep.id)
        a.representation = rep
        return a

    def to_json(self, visible_ids=None):
        return asdict(rewrite_representation(self.representation, visible_ids))

    def __repr__(self):
        return f"Array(id={self.id},\narr={self.arr},\nrepresentation={self.representation})"

    def _repr_javascript_(self):
        return """
        (function(element){
            require(['op_art'], function(op_art) {
                op_art.visualize(element.get(0), %s, [""], 1500, true);
            });
        })(element);
        """ % json.dumps([self.to_json(visible_ids=[self.id])])

@dataclass(frozen=False)  # note frozen due to __setitem__
class CellRepresentation:
    id: Any
    index: Any
    value: Any
    sources: Any = None


@dataclass(frozen=True)
class ArrayRepresentation:
    id: int
    kind: str
    ndim: int
    shape: Any
    cells: Tuple[CellRepresentation]


def _rewrite_sources(source_ids, visible_ids, cell_id_to_sources):
    if source_ids is None:
        return []
    res = []
    for si in source_ids:
        arr_id = int(si.split("_")[0])
        if arr_id in visible_ids:
            res.append(si)
        else:
            res.extend(_rewrite_sources(cell_id_to_sources[si], visible_ids, cell_id_to_sources))
    return res

def rewrite_sources(source_ids, visible_ids):
    """Rewrite sources so none are from invisible arrays"""
    cell_id_to_sources = {}
    for array_id, array in id_to_array.items():
        cells = array.representation.cells
        for cell in cells:
            cell_id_to_sources[cell.id] = cell.sources

    x = _rewrite_sources(source_ids, visible_ids, cell_id_to_sources)
    return x if len(x) > 0 else None

def rewrite_representation(arr_rep, visible_ids=None):
    """Rewrite representation so it doesn't include any invisible arrays"""
    if visible_ids is None:
        return arr_rep
    new_cells = []
    for cell in arr_rep.cells:
        new_sources = rewrite_sources(cell.sources, visible_ids)
        new_cell = CellRepresentation(cell.id, cell.index, cell.value, new_sources)
        new_cells.append(new_cell)
    return ArrayRepresentation(arr_rep.id, arr_rep.kind, arr_rep.ndim, arr_rep.shape, new_cells)

def arrays_to_json(arrays):
    visible_ids = [a.id for a in arrays]
    return json.dumps([a.to_json(visible_ids) for a in arrays], indent=2)

def strings_to_json(strings):
    return json.dumps(strings, indent=2)

def arrays_to_html(arrays, lines, base_url="."):
    require_js_path = f"{base_url}/require.js"
    op_art_css_path = f"{base_url}/op_art.css"
    return """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>op-art</title>
    <script src="%s"></script>
    <link href="%s" rel="stylesheet" />
  </head>

  <body>
    <div id="plot"></div>

    <script>
      require.config({
        baseUrl: "%s",
        paths: {
          d3: 'https://d3js.org/d3.v5.min'
        }
      });
      require(["op_art"], function(op_art) {
        const arrs = %s;
        const lines = %s;

        op_art.visualize("#plot", arrs, lines, 1500, true);
      });
    </script>
  </body>
</html>""" % (require_js_path, op_art_css_path, base_url, arrays_to_json(arrays), strings_to_json(lines))

def write_html(filename, arrs, lines, base_url="."):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(arrays_to_html(arrs, lines, base_url=base_url))


def get_source(fn):
    """Get the source code for a function (excluding declaration and return line."""
    src = inspect.getsource(fn)
    src = src.strip()
    lines = src.split("\n")
    lines = lines[1:-1]  # remove function declaration and return
    src = dedent("\n".join(lines))
    return src.split("\n")


def visualize(*arrays):
    from IPython.display import display, Javascript
    import inspect
    frame = inspect.currentframe()
    frame = frame.f_back # go back one in the stack
    src = inspect.getsource(frame)
    src = src.strip()
    last_line = src.split("\n")[-1].strip()
    # TODO: improve
    vars = last_line[last_line.index("(") + 1:]
    vars = vars[:vars.index(")")]
    vars = [v.strip() for v in vars.split(",")]

    js = """(function(element){
    require(['op_art'], function(op_art) {
        op_art.visualize(element.get(0), %s, %s, 1500, true);
    });
})(element);""" % (arrays_to_json(arrays), vars)

    return display(Javascript(js))

def _broadcast_to(x, shape, arr_id=None):
    arr = np.broadcast_to(x.arr, shape)

    # broadcast_to doesn't change the underlying array, so we can just rewrite the id to get the input id
    new_rep = Array._get_representation(arr, arr_id=arr_id)
    arr = np.copy(arr) # "materialize" array so cell ids in result are unique
    new_rep_materialized = Array._get_representation(arr, arr_id=new_rep.id)
    cells = []
    for cell, cell_mat in zip(new_rep.cells, new_rep_materialized.cells):
        input_id = cell.id.replace(f"{new_rep.id}_", f"{x.id}_")
        cell = CellRepresentation(cell_mat.id, cell.index, cell.value, [input_id])
        cells.append(cell)

    rep = ArrayRepresentation(new_rep.id, arr.dtype.kind, arr.ndim, arr.shape, tuple(cells))
    return Array.from_representation(arr, rep)

def _broadcast_arrays(*arrays):
    shape = np.broadcast_shapes(*[a.arr.shape for a in arrays])

    if all(array.shape == shape for array in arrays):
        return arrays

    return [_broadcast_to(array, shape, array.id) for array in arrays]

# Covers the case where input and output cells correspond directly by iteration order
# (C order), even if they don't have corresponding indexes
def _direct_mapping(input, output):
    x, arr = input, output
    x_index_to_cell = {cell.index: cell for cell in x.representation.cells}
    new_rep = Array._get_representation(arr)
    cells = []
    it = np.nditer(x.arr, flags=["multi_index", "zerosize_ok"], order="C")
    for c, cell in zip(it, new_rep.cells):
        source_id = x_index_to_cell[it.multi_index].id
        cell = CellRepresentation(cell.id, cell.index, cell.value, [source_id])
        cells.append(cell)

    rep = ArrayRepresentation(new_rep.id, arr.dtype.kind, arr.ndim, arr.shape, tuple(cells))
    return Array.from_representation(arr, rep)

# Covers the case where output cells can be mapped back to input cells
# using an index.
# output == input[ind]
def _index_mapping(input, output, ind):
    x, arr = input, output
    new_rep = Array._get_representation(arr)
    cells = []
    for index, cell in zip(ind, new_rep.cells):
        offset = index_to_offset(x.arr, index)
        cell = CellRepresentation(cell.id, cell.index, cell.value, [f"{x.id}_{offset}"])
        cells.append(cell)

    rep = ArrayRepresentation(new_rep.id, arr.dtype.kind, arr.ndim, arr.shape, tuple(cells))
    return Array.from_representation(arr, rep) 

def _elementwise_unary_operation(x, array_op):
    arr = array_op(x.arr)
    return _direct_mapping(x, arr)

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
    new_rep = Array._get_representation(arr)

    # broadcast if necessary, and keep track of ids to map
    x1_broad, x2_broad = _broadcast_arrays(x1, x2)
    id1_mapping = {cell.id: cell.sources[0] for cell in x1_broad.representation.cells if cell.sources is not None}
    id2_mapping = {cell.id: cell.sources[0] for cell in x2_broad.representation.cells if cell.sources is not None}

    cells = []
    for cell in new_rep.cells:
        input_id1 = cell.id.replace(f"{new_rep.id}_", f"{x1.id}_")
        input_id1 = id1_mapping.get(input_id1, input_id1)
        input_id2 = cell.id.replace(f"{new_rep.id}_", f"{x2.id}_")
        input_id2 = id2_mapping.get(input_id2, input_id2)
        cell = CellRepresentation(cell.id, cell.index, cell.value, [input_id1, input_id2])
        cells.append(cell)

    rep = ArrayRepresentation(new_rep.id, arr.dtype.kind, arr.ndim, arr.shape, tuple(cells))
    return Array.from_representation(arr, rep)

def _reduction_operation(x, axis, array_op, py_op=None, fill_value=0):
    arr = array_op(x.arr, axis)
    y = np.full_like(arr, fill_value)

    # create a mapping of (output) index to cell, so we can update the sources for each cell
    new_rep = Array._get_representation(arr)
    index_to_cell = {cell.index: cell for cell in new_rep.cells}

    with np.nditer([x.arr, y],
                        flags=["reduce_ok", "multi_index", "zerosize_ok"],
                        op_flags=[["readonly"], ["readwrite"]],
                        op_axes=[None, axis_to_axeslist(axis, x.ndim)]) as it:
        for a, b in it:
            # find the offset within the output array using the input multi index
            offset = index_to_offset(it.itviews[1], it.multi_index)
            # ... then convert it back to an index in the output array
            index = offset_to_index(y, offset)
            source_offset = f"{index_to_offset(x.arr, it.multi_index)}"
            # update the sources for the cell
            cell = index_to_cell[index]
            sources = cell.sources or []
            sources.append(f"{x.id}_{source_offset}")
            index_to_cell[index] = CellRepresentation(cell.id, cell.index, cell.value, sources)
            if py_op is not None:
                # do the operation (as a sanity check)
                b[...] = py_op(b[...], a)

    if py_op is not None:
        assert_array_equal(y, arr)

    rep = ArrayRepresentation(new_rep.id, arr.dtype.kind, arr.ndim, arr.shape, tuple(index_to_cell.values()))
    return Array.from_representation(arr, rep)